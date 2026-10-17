# -*- coding: utf-8 -*-
"""「污渍 / 字迹」平衡的自动检测。

用真书回答一个只有数据能回答的问题：**透印污渍洗掉了多少，字迹又保留了多少**。
这两个目标天生冲突——污渍洗得越狠，低对比的字迹就越容易被一并漂白。

三个指标（均为真书逐页统计后汇总）：

  1. 字芯浓度 —— 源图深色笔画（< 100 灰阶）在成品中的平均灰度。越低越黑；
     过高说明字被漂白。
  2. 晕边保真 —— 源图中灰晕边（110-160 灰阶、距深色墨迹 ≤ 8px）在成品中的
     平均灰度。过高说明洇墨晕边被漂白（即「文字偏浅」）。
  3. 污渍清洁 —— 源图中灰、且远离深色墨迹（> 20px）的孤立像素（透印污渍），
     成品中被漂白（> 220）的占比。过低说明污渍没洗干净。

判定（`BalanceTargets`）任一不满足即判失败。运行方式：

    python scripts/run_balance.py              # 独立运行，写 reports/balance.json
    pytest -m slow                              # 作为测试跑（pytest.ini 默认跳过）
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pymupdf

from classicsclean.algorithms.masks import compute_masks
from classicsclean.config import DEFAULT_PARAMS, Params
from classicsclean.imaging import page_image
from classicsclean.pipeline.page import level_page

ROOT = Path(__file__).resolve().parents[2]
SAMPLE_DIR = ROOT / "samples"
REPORTS_DIR = ROOT / "reports"
BALANCE_JSON = REPORTS_DIR / "balance.json"

# 距离度量用 cv2（开发/测试环境有 cv2；成品 exe 不依赖它）
_KERNEL = np.ones((3, 3), np.uint8)


@dataclass(frozen=True)
class BalanceTargets:
    """平衡阈值：任一不满足即判失败。

    仅对「纸面可辨」的页（纸面中位灰度 ≥ NORMAL_PAPER_MIN）生效：深色扉页 /
    封面等低对比页里，淡墨标题与灰背景只差几个灰阶，「字迹 vs 污渍」无法区分，
    单独统计报告、不参与判失败。
    """

    ink_core_mean_max: float = 130.0   # 字芯输出均值上限（越低越黑）
    halo_mean_max: float = 215.0       # 晕边输出均值上限（过高 = 文字偏浅）
    stain_clean_min: float = 90.0      # 污渍漂白率下限（>220 占比，%）
    ink_core_diff_max: int = 0         # 红线：字芯逐像素等于基线（算法差必须为 0）


NORMAL_PAPER_MIN = 150.0
"""「纸面可辨」的最低纸面中位灰度：低于此值视为低对比页（深色封面等），
其淡墨内容与背景几乎同灰度，不计入平衡判定。"""


def _ink_distance(ink: np.ndarray) -> np.ndarray:
    """到深色墨迹的距离（像素），用于区分「晕边」与「孤立污渍」。"""
    return cv2.distanceTransform((~ink).astype(np.uint8), cv2.DIST_L2, 3)


def _zones(masks) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回 (字芯, 晕边, 污渍) 三个布尔掩膜。"""
    gray = masks.gray
    dist = _ink_distance(masks.ink)
    ink_core = gray < 100.0
    halo = (gray >= 110.0) & (gray < 160.0) & (dist <= 8.0) & masks.paper
    stain = (gray >= 110.0) & (gray < 160.0) & (dist > 20.0) & masks.paper
    return ink_core, halo, stain


def run(source: Path, params: Params = DEFAULT_PARAMS) -> Tuple[List[Dict], Dict]:
    """跑全书平衡检测，返回 (逐页数据, 汇总)。"""
    src_doc = pymupdf.open(str(source))
    n = src_doc.page_count
    rows: List[Dict] = []
    t0 = time.time()
    try:
        for i in range(n):
            rgb = page_image(src_doc, i)
            masks = compute_masks(rgb, params)
            new, _, info = level_page(rgb, params, pre=masks)
            base, _, _ = level_page(rgb, Params(gamma=1.0, wp_mode="whitest",
                                                 text_guard=1), pre=masks)
            ink_core, halo, stain = _zones(masks)
            core = cv2.erode(masks.ink.astype(np.uint8), _KERNEL, iterations=2) > 0
            row = {
                "页": i + 1,
                "白场模式": info["mode"],
                "纸面中位": round(float(np.median(masks.gray[masks.paper])), 1),
                "字芯输出均值": round(float(new[ink_core].mean()), 1) if ink_core.any() else None,
                "晕边输出均值": round(float(new[halo].mean()), 1) if halo.any() else None,
                "污渍漂白率%": round(100 * float((new[stain] > 220).mean()), 1) if stain.any() else None,
                "污渍像素数": int(stain.sum()),
                "字核心算法差": int(np.abs(new[core].astype(np.int16)
                                         - base[core].astype(np.int16)).max()) if core.any() else 0,
            }
            rows.append(row)
            print(f"p{i + 1:02d} 纸面 {row['纸面中位']:.0f} | 字芯 {row['字芯输出均值']} | 晕边 {row['晕边输出均值']} | "
                  f"污渍漂白率 {row['污渍漂白率%']}%（污渍 {row['污渍像素数']}px）", flush=True)
    finally:
        src_doc.close()
    return rows, summarize(rows, source, time.time() - t0)


def summarize(rows: List[Dict], source: Path, elapsed: float) -> Dict:
    normal = [r for r in rows if r["纸面中位"] >= NORMAL_PAPER_MIN]
    low = [r["页"] for r in rows if r["纸面中位"] < NORMAL_PAPER_MIN]

    def vals(key):
        return [r[key] for r in normal if r[key] is not None]

    # 字芯 / 晕边只在「纸面可辨」页上判（低对比页的淡墨与背景无法区分）
    ink = vals("字芯输出均值")
    halo = vals("晕边输出均值")
    stain = [r["污渍漂白率%"] for r in rows if r["污渍漂白率%"] is not None]
    core_diff = [r["字核心算法差"] for r in rows]
    return {
        "源文件": source.name,
        "页数": len(rows),
        "纸面可辨页数(参与判定)": len(normal),
        "低对比页(仅报告，不判定)": low or "无",
        "字芯输出均值(平均/最差)": [round(float(np.mean(ink)), 1), max(ink)],
        "晕边输出均值(平均/最差)": [round(float(np.mean(halo)), 1), max(halo)],
        "污渍漂白率%(平均/最差)": [round(float(np.mean(stain)), 1), min(stain)],
        "字核心算法差最大": max(core_diff),
        "耗时秒": round(elapsed, 1),
    }


def check(summary: Dict, targets: BalanceTargets = None) -> List[str]:
    """按阈值检查汇总，返回失败原因列表（空 = 通过）。"""
    t = targets or BalanceTargets()
    fails: List[str] = []
    if summary["字芯输出均值(平均/最差)"][1] > t.ink_core_mean_max:
        fails.append(f"字芯被漂白：最差均值 {summary['字芯输出均值(平均/最差)'][1]} > {t.ink_core_mean_max}")
    if summary["晕边输出均值(平均/最差)"][1] > t.halo_mean_max:
        fails.append(f"晕边被漂白（文字偏浅）：最差均值 {summary['晕边输出均值(平均/最差)'][1]} > {t.halo_mean_max}")
    if summary["污渍漂白率%(平均/最差)"][1] < t.stain_clean_min:
        fails.append(f"污渍没洗干净：最差漂白率 {summary['污渍漂白率%(平均/最差)'][1]}% < {t.stain_clean_min}%")
    if summary["字核心算法差最大"] > t.ink_core_diff_max:
        fails.append(f"字核心被改动：算法差 {summary['字核心算法差最大']} > 0")
    return fails


def find_source(sample_dir: Path = SAMPLE_DIR):
    pdfs = sorted(p for p in sample_dir.glob("*.pdf") if "_清洁版" not in p.stem)
    return pdfs[0] if pdfs else None


def write_report(rows: List[Dict], summary: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"summary": summary, "pages": rows},
                               ensure_ascii=False, indent=2), encoding="utf-8")
