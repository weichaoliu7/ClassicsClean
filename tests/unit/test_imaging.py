# -*- coding: utf-8 -*-
"""基础层单元测试：PDF ↔ 位图、灰度化、JPEG2000 编解码。"""
import numpy as np
import pymupdf
import pytest

from classicsclean.imaging import (
    FALLBACK_DPI,
    decode_jp2,
    encode_jp2,
    page_image,
    page_rect,
    to_gray,
)


def test_to_gray_uses_rec601():
    """灰度化必须采用 Rec.601 权重。"""
    rgb = np.zeros((2, 2, 3), np.float32)
    rgb[0, 0] = (255, 0, 0)
    rgb[0, 1] = (0, 255, 0)
    rgb[1, 0] = (0, 0, 255)
    g = to_gray(rgb)
    assert abs(g[0, 0] - 0.299 * 255) < 0.5
    assert abs(g[0, 1] - 0.587 * 255) < 0.5
    assert abs(g[1, 0] - 0.114 * 255) < 0.5


def test_to_gray_passthrough_2d():
    """已经是灰度时原样返回，不重复计算。"""
    arr = np.full((4, 4), 128.0, np.float32)
    assert to_gray(arr) is arr


def test_jp2_roundtrip_is_nearly_lossless():
    """52 dB 编码下平坦区域的往返误差应极小。"""
    probe = np.full((96, 96), 200, np.uint8)
    back = decode_jp2(encode_jp2(probe, 52.0), probe.shape)
    assert back.shape == probe.shape
    assert int(np.abs(back.astype(int) - probe.astype(int)).max()) <= 2


def test_jp2_quality_affects_size():
    """质量档位必须真实生效（高 dB -> 更大码流）。"""
    rng = np.random.default_rng(0)
    img = (128 + 60 * rng.normal(size=(256, 256))).clip(0, 255).astype(np.uint8)
    low = len(encode_jp2(img, 40.0))
    high = len(encode_jp2(img, 60.0))
    assert high > low, "高保真档位的码流必须更大"


def test_jp2_survives_tiny_images():
    """极小的图不会因分解层数过高而编码失败。"""
    tiny = np.full((8, 8), 100, np.uint8)
    assert len(encode_jp2(tiny, 52.0, num_resolutions=6)) > 0
    assert decode_jp2(encode_jp2(tiny, 52.0)).shape == (8, 8)


def test_page_image_reads_embedded_image(scan_pdf):
    """优先取内嵌主图像，尺寸与原图一致。"""
    with pymupdf.open(scan_pdf) as doc:
        rgb = page_image(doc, 0)
        assert rgb.ndim == 3 and rgb.shape[2] == 3
        assert rgb.dtype == np.uint8
        assert rgb.shape[:2] == (880, 640)


def test_page_image_falls_back_to_render(tmp_path):
    """纯文本页（无内嵌图像）必须退化为整页渲染，而不是报错。"""
    doc = pymupdf.open()
    page = doc.new_page(width=300, height=400)
    page.insert_text((72, 120), "ClassicsClean", fontsize=24)
    path = tmp_path / "text-only.pdf"
    doc.save(str(path))
    doc.close()
    with pymupdf.open(str(path)) as d:
        rgb = page_image(d, 0)
    assert rgb.ndim == 3 and rgb.shape[0] > 0
    assert FALLBACK_DPI == 200


def test_page_rect_matches_source(scan_pdf):
    """页面几何原样返回，保证输出版式不变。"""
    with pymupdf.open(scan_pdf) as doc:
        rect = page_rect(doc, 1)
        assert (rect.width, rect.height) == (doc[1].rect.width, doc[1].rect.height)
