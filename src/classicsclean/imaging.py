# -*- coding: utf-8 -*-
"""【第 1 层 · 基础】位图提取、灰度化与 JPEG2000 编解码。

输入：pymupdf 文档 + 页号，或 uint8 灰度数组
输出：RGB / 灰度 numpy 数组，或 JPEG2000 字节流
边界：只做「PDF ↔ numpy 位图」的搬运与编码，**不含任何图像算法**
"""
from __future__ import annotations

import io
import math
from typing import Optional, Tuple

import numpy as np
import pymupdf
from PIL import Image

from .morphology import area_resize

__all__ = [
    "page_image",
    "to_gray",
    "encode_jp2",
    "decode_jp2",
    "page_rect",
    "FALLBACK_DPI",
]

FALLBACK_DPI = 200
"""页面没有内嵌图像（纯文本页 / 矢量页）时的渲染 dpi。"""


def page_image(doc: "pymupdf.Document", index: int) -> np.ndarray:
    """取出第 index 页的主位图，返回 RGB uint8 数组。

    优先取该页**面积最大的内嵌图像**：扫描古籍 PDF 常在图像之上叠加 OCG 馆藏章
    水印层，直接渲染整页会把水印混进来。没有内嵌图像时退化为整页渲染。
    """
    page = doc[index]
    best = None
    for img in page.get_images(full=True):
        area = img[2] * img[3]
        if best is None or area > best[0]:
            best = (area, img[0])
    if best is None:
        pix = page.get_pixmap(dpi=FALLBACK_DPI, colorspace=pymupdf.csRGB)
        return np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, 3)[:, :, :3]
    pix = pymupdf.Pixmap(doc, best[1])
    if pix.n > 3:
        pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
    if pix.n == 1:
        pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
    arr = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)
    return arr[:, :, :3].copy()


def to_gray(rgb: np.ndarray) -> np.ndarray:
    """RGB -> float32 灰度（Rec.601 权重）。"""
    arr = np.asarray(rgb, dtype=np.float32)
    if arr.ndim == 2:
        return arr
    return 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]


def page_rect(doc: "pymupdf.Document", index: int) -> "pymupdf.Rect":
    """页面几何（输出 PDF 完全沿用，保证版式不变）。"""
    return doc[index].rect


def _safe_resolutions(width: int, height: int, wanted: int) -> int:
    """JPEG2000 分解层数不能超过图像尺寸允许的上限。"""
    limit = int(math.floor(math.log2(max(2, min(width, height))))) - 1
    return int(max(1, min(int(wanted), max(1, limit))))


def encode_jp2(gray: np.ndarray, db: float, num_resolutions: int = 6) -> bytes:
    """灰度图 -> JPEG2000 码流（不可逆、以 dB 指定质量）。"""
    arr = np.asarray(gray)
    if arr.ndim == 3:
        arr = to_gray(arr)
    pil = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).convert("L")
    res = _safe_resolutions(pil.width, pil.height, num_resolutions)
    buf = io.BytesIO()
    pil.save(
        buf,
        format="JPEG2000",
        irreversible=True,
        quality_mode="dB",
        quality_layers=[float(db)],
        num_resolutions=res,
    )
    return buf.getvalue()


def decode_jp2(data: bytes, shape: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """JPEG2000 码流 -> 灰度图（校验/测试用；尺寸不一致时按面积缩放对齐）。"""
    arr = np.array(Image.open(io.BytesIO(data)).convert("L"))
    if shape is not None and arr.shape != shape:
        arr = area_resize(arr, shape[1], shape[0])
    return arr
