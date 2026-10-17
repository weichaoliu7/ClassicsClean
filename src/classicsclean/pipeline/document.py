# -*- coding: utf-8 -*-
"""【第 3 层 · 编排】整本流水线：读源 PDF → 逐页处理 → 写目标 PDF → 可选回读校验。

输入：源 PDF 路径、目标 PDF 路径、Params、进度回调
输出：Result（逐页报告 + 体积 + 耗时 + 可选校验行）
边界：这是 GUI / CLI / 测试共用的**唯一 IO 边界**，任何调用方都不应自己拼装 PDF
"""
from __future__ import annotations

import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import pikepdf
import pymupdf

from ..config import DEFAULT_PARAMS, Params
from ..imaging import encode_jp2, page_image, page_rect
from ..morphology import area_resize
from .page import level_page

__all__ = ["PageReport", "ValidationRow", "Result", "process_pdf", "validate_output"]


@dataclass
class PageReport:
    """单页处理结果。"""

    index: int
    width: int
    height: int
    payload: int            # JPEG2000 码流字节数
    info: Dict = field(default_factory=dict)


@dataclass
class ValidationRow:
    """单页回读校验结果。"""

    page: int
    psnr: float             # 与算法参考图相比的峰值信噪比（dB）
    maxerr: int             # 最大单像素误差
    margin_dirty: float     # 白边区域被压暗（<250）的比例
    ink_ratio: float        # 输出图中深墨像素占比


@dataclass
class Result:
    """一次完整处理任务的结果。"""

    src: str
    dst: str
    pages: List[PageReport]
    elapsed: float
    output_bytes: int
    validation: Optional[List[ValidationRow]] = None

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def image_ratio(self) -> float:
        """输出体积 / 输入体积。"""
        try:
            return self.output_bytes / max(1, os.path.getsize(self.src))
        except OSError:
            return float("nan")


def _write_pdf(pages: List[PageReport], payloads: List[Path], rects, dst: str, title: str) -> str:
    """把处理后的页序列写成 PDF（JPEG2000 灰度，页面几何完全沿用源文档）。

    payloads 是临时目录里的 .jp2 文件列表 —— 大部头书籍若把全部码流留在内存里，
    低配电脑会吃不消，所以逐页落盘、装配时再读回来。
    """
    with pikepdf.new() as pdf:
        pdf.Root.PageMode = pikepdf.Name.UseNone
        try:
            with pdf.open_metadata() as meta:
                meta["dc:title"] = title
                meta["dc:creator"] = ["ClassicsClean 古籍扫描件白底清洁工具"]
        except Exception:      # 元数据不可写时不影响正文
            pass
        for report, payload_path, rect in zip(pages, payloads, rects):
            img = pikepdf.Stream(pdf, payload_path.read_bytes())
            img.Type = pikepdf.Name.XObject
            img.Subtype = pikepdf.Name.Image
            img.Width = report.width
            img.Height = report.height
            img.BitsPerComponent = 8
            img.ColorSpace = pikepdf.Name.DeviceGray
            img.Filter = pikepdf.Name.JPXDecode
            content = b"q %f 0 0 %f 0 0 cm /Im0 Do Q" % (rect.width, rect.height)
            page = pikepdf.Dictionary(
                Type=pikepdf.Name.Page,
                MediaBox=[rect.x0, rect.y0, rect.x1, rect.y1],
                Resources=pikepdf.Dictionary(XObject=pikepdf.Dictionary({"/Im0": img})),
                Contents=pdf.make_stream(content),
            )
            pdf.pages.append(pikepdf.Page(page))
        try:
            pdf.save(dst)
        except PermissionError:
            # 目标被预览器/阅读器占用时另存，绝不覆盖失败
            dst = str(Path(dst).with_name(f"{Path(dst).stem}_新.pdf"))
            pdf.save(dst)
    return dst


def process_pdf(
    src: str,
    dst: str,
    params: Optional[Params] = None,
    progress: Optional[Callable[[int, int, str], None]] = None,
    validate: Optional[bool] = None,
) -> Result:
    """处理整本 PDF。

    :param src:      源 PDF 路径
    :param dst:      目标 PDF 路径（父目录不存在时自动创建）
    :param params:   算法参数，默认 DEFAULT_PARAMS
    :param progress: 进度回调 (已完成页数, 总页数, 说明文字)
    :param validate: 是否回读校验，None 时沿用 params.validate
    """
    params = params or DEFAULT_PARAMS
    validate = params.validate if validate is None else validate
    src_path = Path(src)
    dst_path = Path(dst)
    if not src_path.is_file():
        raise FileNotFoundError(f"源文件不存在：{src_path}")
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    reports: List[PageReport] = []
    payloads: List[Path] = []
    rects = []

    # 逐页码流落临时目录：内存里只留当前页，避免大部头书籍在低配电脑上爆内存
    with tempfile.TemporaryDirectory(prefix="cclean_") as tmpdir:
        with pymupdf.open(str(src_path)) as doc:
            total = doc.page_count
            for i in range(total):
                rgb = page_image(doc, i)
                out, margin, info = level_page(rgb, params)
                data = encode_jp2(out, params.jp2_db, params.jp2_num_resolutions)
                rects.append(page_rect(doc, i))
                payload_path = Path(tmpdir) / f"{i:06d}.jp2"
                payload_path.write_bytes(data)
                payloads.append(payload_path)
                reports.append(
                    PageReport(
                        index=i,
                        width=int(out.shape[1]),
                        height=int(out.shape[0]),
                        payload=len(data),
                        info=info,
                    )
                )
                if progress:
                    progress(
                        i + 1,
                        total,
                        f"第 {i + 1}/{total} 页  白场 {info['Wp']:.1f}  纸面亮度 {info['paper_level']:.0f}"
                        f"  {info['mode']}  破洞 补{info['filled']}/跳{info['skipped']}",
                    )
                del out, rgb, data

        written = _write_pdf(reports, payloads, rects, str(dst_path),
                             f"{src_path.stem}（白底清洁版）")
        elapsed = time.time() - t0

        result = Result(
            src=str(src_path),
            dst=written,
            pages=reports,
            elapsed=elapsed,
            output_bytes=os.path.getsize(written),
        )
        if validate:
            if progress:
                progress(total, total, "出片完成，开始回读校验…")
            result.validation = validate_output(written, str(src_path), params)
    return result


def validate_output(dst: str, src: str, params: Optional[Params] = None) -> List[ValidationRow]:
    """重新渲染输出 PDF，与「重新计算」的算法参考图逐页比对。

    这一步同时验证了算法的确定性与 JPEG2000 封装的正确性：
    若编码写错（如色彩空间/滤波参数），PSNR 会断崖式下跌。
    """
    params = params or DEFAULT_PARAMS
    rows: List[ValidationRow] = []
    with pymupdf.open(src) as src_doc, pymupdf.open(dst) as dst_doc:
        assert dst_doc.page_count == src_doc.page_count, "页数不一致"
        for i in range(src_doc.page_count):
            ref, margin, _ = level_page(page_image(src_doc, i), params)
            rect = src_doc[i].rect
            dpi = int(round(72 * ref.shape[1] / rect.width))
            pix = dst_doc[i].get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY)
            rendered = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width)
            rr = area_resize(rendered, ref.shape[1], ref.shape[0])
            err = np.abs(rr.astype(np.int16) - ref.astype(np.int16))
            mse = float((err.astype(np.float64) ** 2).mean())
            psnr = 10 * np.log10(255 ** 2 / max(mse, 1e-9))
            rows.append(
                ValidationRow(
                    page=i + 1,
                    psnr=psnr,
                    maxerr=int(err.max()),
                    margin_dirty=float((rr[margin] < 250).mean()) if margin.any() else 0.0,
                    ink_ratio=float((ref < 64).mean()),
                )
            )
            del ref
    return rows
