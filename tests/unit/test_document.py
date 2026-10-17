# -*- coding: utf-8 -*-
"""整本流水线（process_pdf / validate_output）测试：合成 PDF 走完整链路。"""
import os

import pikepdf
import pymupdf

from classicsclean.pipeline.document import process_pdf, validate_output


def test_process_pdf_creates_output(scan_pdf, tmp_path):
    dst = tmp_path / "out.pdf"
    result = process_pdf(scan_pdf, str(dst), progress=lambda d, t, m: None)
    assert os.path.isfile(result.dst)
    assert result.page_count == 3
    assert result.output_bytes > 0
    assert result.elapsed > 0


def test_output_geometry_matches_source(scan_pdf, tmp_path):
    """输出页面尺寸必须与源文档逐页一致（版式不能变）。"""
    dst = tmp_path / "out.pdf"
    process_pdf(scan_pdf, str(dst))
    with pymupdf.open(scan_pdf) as src, pymupdf.open(str(dst)) as out:
        assert src.page_count == out.page_count
        for i in range(src.page_count):
            a, b = src[i].rect, out[i].rect
            assert abs(a.width - b.width) < 0.5 and abs(a.height - b.height) < 0.5


def test_output_is_grayscale_jpeg2000(scan_pdf, tmp_path):
    """每页必须是单张 DeviceGray + JPXDecode 图像（8bit 灰度 JPEG2000）。"""
    dst = tmp_path / "out.pdf"
    process_pdf(scan_pdf, str(dst))
    with pikepdf.open(str(dst)) as pdf:
        for page in pdf.pages:
            xobjects = page.Resources.XObject
            assert len(xobjects.keys()) == 1, "每页应只嵌一张处理后的主图像"
            img = xobjects["/Im0"]
            assert str(img.Filter) == "/JPXDecode", "必须使用 JPEG2000 编码"
            assert str(img.ColorSpace) == "/DeviceGray", "输出必须是灰度"
            assert int(img.BitsPerComponent) == 8


def test_validate_output_quality(scan_pdf, tmp_path):
    """回读校验：合成页 52dB 编码下 PSNR 应显著高于 35dB。"""
    dst = tmp_path / "out.pdf"
    process_pdf(scan_pdf, str(dst))
    rows = validate_output(str(dst), scan_pdf)
    assert len(rows) == 3
    for row in rows:
        assert row.psnr > 35.0, f"第 {row.page} 页 PSNR 偏低：{row.psnr:.2f} dB"
        assert row.margin_dirty < 0.02, f"第 {row.page} 页白边被污染：{row.margin_dirty:.4f}"


def test_process_pdf_with_validation_flag(scan_pdf, tmp_path):
    """validate=True 时结果里必须带回逐页校验数据。"""
    dst = tmp_path / "out.pdf"
    result = process_pdf(scan_pdf, str(dst), validate=True)
    assert result.validation is not None and len(result.validation) == 3


def test_output_dir_auto_created(scan_pdf, tmp_path):
    """输出目录不存在时自动创建。"""
    dst = tmp_path / "nested" / "deeper" / "out.pdf"
    result = process_pdf(scan_pdf, str(dst))
    assert os.path.isfile(result.dst)
