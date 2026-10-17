# -*- coding: utf-8 -*-
"""全书终检引擎：用真书回答三个统计问题。

与单元测试的分工：
  * 单元测试（tests/unit/）用**合成页**做精确断言，秒级完成；
  * 本模块用**真书**回答三个统计问题，分钟级：
      A. 背景干净了吗 —— 纸面（排除白边 / 墨迹 / 破洞）是否接近纯白
      B. 字被改动了吗 —— 与「基线白场 + 线性映射」逐像素比对，字核心差必须为 0
      C. 成品 PDF 对吗 —— 重新渲染最终 PDF，与算法参考图比对（PSNR / 最大误差）

运行方式：
    python scripts/run_acceptance.py            # 独立运行，写报告
    pytest -m slow                              # 作为测试跑（pytest.ini 默认跳过）
"""
from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pymupdf

from classicsclean.algorithms.masks import compute_masks
from classicsclean.config import DEFAULT_PARAMS, Params
from classicsclean.imaging import decode_jp2, encode_jp2, page_image
from classicsclean.pipeline.document import process_pdf
from classicsclean.pipeline.page import level_page, text_guard_mask

ROOT = Path(__file__).resolve().parents[2]
SAMPLE_DIR = ROOT / "samples"
REPORTS_DIR = ROOT / "reports"
ACCEPTANCE_CSV = REPORTS_DIR / "acceptance-pages.csv"
ACCEPTANCE_JSON = REPORTS_DIR / "acceptance.json"

OUTPUT_SUFFIX = "_清洁版"
"""成品样本的后缀，与 CLI 默认输出命名保持一致。"""

BASELINE_PARAMS = Params(gamma=1.0, wp_mode="whitest", text_guard=1)
"""对照基线：早期「全局最白区白点 + 线性映射」方案，用来证明新算法没有动字。"""


@dataclass(frozen=True)
class AcceptanceTargets:
    """终检阈值：任一不满足即判失败。"""

    paper_lt240_max: float = 15.0        # 纸面 <240 的占比均值上限（%）
    ink_core_algo_diff_max: int = 0      # 字核心算法差最大值（红线：必须为 0）
    ink_core_codec_diff_max: int = 20    # 字核心编码差最大值
    edge_codec_diff_max: int = 40        # 抗锯齿边缘编码差最大值
    psnr_min: float = 40.0               # 成品 PDF 与算法参考图的最低 PSNR（dB）
    render_max_err_max: int = 40         # 成品 PDF 单像素最大误差
    margin_dirty_max: float = 1.0        # 白边污染率上限（%）


def find_source(sample_dir: Path = SAMPLE_DIR) -> Optional[Path]:
    """在 samples/ 中找到源 PDF（不含成品后缀的那一个）。"""
    if not sample_dir.is_dir():
        return None
    pdfs = sorted(p for p in sample_dir.glob("*.pdf") if OUTPUT_SUFFIX not in p.stem)
    return pdfs[0] if pdfs else None


def ensure_output(source: Path, db: float = DEFAULT_PARAMS.jp2_db) -> Path:
    """返回源 PDF 对应的成品 PDF；不存在时用当前代码现场生成，保证样本与代码同步。"""
    expected = source.with_name(f"{source.stem}{OUTPUT_SUFFIX}.pdf")
    if expected.is_file():
        return expected
    result = process_pdf(str(source), str(expected), DEFAULT_PARAMS.with_db(db))
    return Path(result.dst)


def _clean_paper(masks, params: Params) -> np.ndarray:
    """纸面统计区：纸面 且 排除文字保护区（墨迹含低对比晕边）与破洞。

    木版印刷的洇墨晕边是墨迹的一部分（理应保留其浓度，而非漂白），不能计入
    「纸面洁净度」；用文字保护掩膜（墨迹 ∪ 晕边）整体排除，比固定 3px 的
    墨迹膨胀更贴合实际墨迹范围。"""
    guard = text_guard_mask(masks.gray, masks, params)
    dil = cv2.dilate(guard.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=3)
    return masks.paper & ~(dil > 0) & ~masks.holes


def _ink_zones(masks) -> Tuple[np.ndarray, np.ndarray]:
    """字核心（腐蚀 2 次）与抗锯齿边缘（膨胀 1 次后去掉字身的环）。"""
    k3 = np.ones((3, 3), np.uint8)
    core = cv2.erode(masks.ink.astype(np.uint8), k3, iterations=2) > 0
    dilated = cv2.dilate(masks.ink.astype(np.uint8), k3, iterations=1) > 0
    return core, dilated & (~masks.ink) & (~masks.margin)


def _max_diff(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> int:
    if not mask.any():
        return 0
    return int(np.abs(a[mask].astype(np.int16) - b[mask].astype(np.int16)).max())


def _mean_diff(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    if not mask.any():
        return 0.0
    return float(np.abs(a[mask].astype(np.int16) - b[mask].astype(np.int16)).mean())


def run(
    source: Path,
    output: Optional[Path] = None,
    db: float = DEFAULT_PARAMS.jp2_db,
) -> Tuple[List[Dict], Dict]:
    """跑全书终检，返回 (逐页数据, 汇总字典)。"""
    params = DEFAULT_PARAMS.with_db(db)
    output = Path(output) if output else ensure_output(source, db)

    src_doc = pymupdf.open(str(source))
    out_doc = pymupdf.open(str(output)) if output.is_file() else None
    n = src_doc.page_count
    if out_doc is not None and out_doc.page_count != n:
        print(f"[警告] 成品页数 {out_doc.page_count} != 源页数 {n}，跳过成品回读")
        out_doc.close()
        out_doc = None

    rows: List[Dict] = []
    t0 = time.time()
    try:
        for i in range(n):
            rgb = page_image(src_doc, i)
            masks = compute_masks(rgb, params)
            new, _, info = level_page(rgb, params, pre=masks)
            base, _, _ = level_page(rgb, BASELINE_PARAMS, pre=masks)

            clean = _clean_paper(masks, params)
            core, edge = _ink_zones(masks)
            v = new[clean].astype(np.float32)

            # 同等编码质量下比对：这是「字有没有被动」的唯一严格判据
            e_new = decode_jp2(encode_jp2(new, db), new.shape)
            e_base = decode_jp2(encode_jp2(base, db), base.shape)

            row: Dict = {
                "页": i + 1,
                "宽": info["width"],
                "高": info["height"],
                "白场模式": info["mode"],
                "白点": round(info["Wp"], 2),
                "纸面亮度": round(info["paper_level"], 2),
                "补洞": info["filled"],
                "碰字跳过": info["skipped"],
                "源纸面中位": round(float(np.median(masks.gray[clean])), 1),
                "成品纸面中位": round(float(np.median(v)), 1),
                "成品纸面均值": round(float(v.mean()), 1),
                "纸面<230占比%": round(100 * float((v < 230).mean()), 2),
                "纸面<240占比%": round(100 * float((v < 240).mean()), 2),
                "纸面<250占比%": round(100 * float((v < 250).mean()), 2),
                "字核心算法差": _max_diff(new, base, core),
                "字核心编码差": _max_diff(e_new, e_base, core),
                "字核心编码均差": round(_mean_diff(e_new, e_base, core), 3),
                "边缘编码差": _max_diff(e_new, e_base, edge),
            }

            if out_doc is not None:                      # C. 成品 PDF 回读
                rect = src_doc[i].rect
                out_rect = out_doc[i].rect
                dpi = int(round(72 * new.shape[1] / rect.width))
                pix = out_doc[i].get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY)
                rendered = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width)
                rr = cv2.resize(rendered, (new.shape[1], new.shape[0]), interpolation=cv2.INTER_AREA)
                err = np.abs(rr.astype(np.int16) - new.astype(np.int16))
                mse = float((err.astype(np.float64) ** 2).mean())
                row["成品PSNR(dB)"] = round(10 * np.log10(255 ** 2 / max(mse, 1e-9)), 3)
                row["成品最大误差"] = int(err.max())
                row["白边污染%"] = (round(100 * float((rr[masks.margin] < 250).mean()), 4)
                                    if masks.margin.any() else 0.0)
                row["页面尺寸一致"] = "是" if abs(rect.width - out_rect.width) < 0.5 \
                    and abs(rect.height - out_rect.height) < 0.5 else "否"

            rows.append(row)
            print(f"p{i + 1:02d} 纸面 {row['源纸面中位']}->{row['成品纸面中位']} "
                  f"<240={row['纸面<240占比%']:5.2f}% | 字核心差(算法/编码)={row['字核心算法差']}/"
                  f"{row['字核心编码差']} | "
                  + (f"PSNR={row['成品PSNR(dB)']:.2f}dB" if out_doc is not None else "无成品"),
                  flush=True)
            del rgb, new, base, e_new, e_base
    finally:
        src_doc.close()
        if out_doc is not None:
            out_doc.close()

    return rows, summarize(rows, source, output, db, time.time() - t0)


def summarize(rows: List[Dict], source: Path, output: Optional[Path], db: float,
              elapsed: float) -> Dict:
    """把逐页数据汇总成一份可断言、可写报告的字典。"""

    def col(key: str) -> List:
        return [r[key] for r in rows if key in r]

    summary: Dict = {
        "源文件": source.name,
        "成品文件": output.name if output and output.is_file() else None,
        "页数": len(rows),
        "编码质量dB": db,
        "源体积MB": round(source.stat().st_size / 1048576, 2),
        "成品体积MB": (round(output.stat().st_size / 1048576, 2)
                     if output and output.is_file() else None),
        "源纸面中位": float(np.median(col("源纸面中位"))),
        "成品纸面中位(min/中位/max)": [min(col("成品纸面中位")),
                                       float(np.median(col("成品纸面中位"))),
                                       max(col("成品纸面中位"))],
        "纸面<240占比%均值": round(float(np.mean(col("纸面<240占比%"))), 2),
        "纸面<250占比%均值": round(float(np.mean(col("纸面<250占比%"))), 2),
        "字核心算法差最大": max(col("字核心算法差")),
        "字核心编码差最大": max(col("字核心编码差")),
        "字核心编码均差均值": round(float(np.mean(col("字核心编码均差"))), 3),
        "边缘编码差最大": max(col("边缘编码差")),
        "补洞合计": int(sum(col("补洞"))),
        "碰字跳过合计": int(sum(col("碰字跳过"))),
        "耗时秒": round(elapsed, 1),
    }
    if rows and "成品PSNR(dB)" in rows[0]:
        summary["成品PSNR最低dB"] = round(min(col("成品PSNR(dB)")), 3)
        summary["成品最大误差"] = max(col("成品最大误差"))
        summary["白边污染%最大"] = max(col("白边污染%"))
        summary["页面尺寸全部一致"] = all(r["页面尺寸一致"] == "是" for r in rows)
    return summary


def check(summary: Dict, targets: Optional[AcceptanceTargets] = None) -> List[str]:
    """按阈值检查汇总数据，返回失败原因列表（空列表 = 通过）。"""
    t = targets or AcceptanceTargets()
    fails: List[str] = []
    if summary["字核心算法差最大"] > t.ink_core_algo_diff_max:
        fails.append(f"字核心被改动：最大算法差 {summary['字核心算法差最大']} > {t.ink_core_algo_diff_max}")
    if summary["字核心编码差最大"] > t.ink_core_codec_diff_max:
        fails.append(f"字核心编码噪声过大：{summary['字核心编码差最大']} > {t.ink_core_codec_diff_max}")
    if summary["边缘编码差最大"] > t.edge_codec_diff_max:
        fails.append(f"抗锯齿边缘编码噪声过大：{summary['边缘编码差最大']} > {t.edge_codec_diff_max}")
    if summary["纸面<240占比%均值"] >= t.paper_lt240_max:
        fails.append(f"纸面洁净度不达标：<240 占比 {summary['纸面<240占比%均值']}% >= {t.paper_lt240_max}%")
    if "成品PSNR最低dB" in summary:
        if summary["成品PSNR最低dB"] <= t.psnr_min:
            fails.append(f"成品 PDF 与算法不一致：最低 PSNR {summary['成品PSNR最低dB']} dB <= {t.psnr_min}")
        if summary["成品最大误差"] > t.render_max_err_max:
            fails.append(f"成品回读误差过大：{summary['成品最大误差']} > {t.render_max_err_max}")
        if summary["白边污染%最大"] > t.margin_dirty_max:
            fails.append(f"白边被污染：{summary['白边污染%最大']}% > {t.margin_dirty_max}%")
        if not summary["页面尺寸全部一致"]:
            fails.append("成品页面尺寸与源文档不一致")
    return fails


def write_reports(rows: List[Dict], summary: Dict, csv_path: Path,
                  json_path: Optional[Path] = None) -> None:
    """把逐页数据与汇总写到报告目录。"""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    if json_path:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(
            json.dumps({"summary": summary, "pages": rows}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
