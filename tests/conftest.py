# -*- coding: utf-8 -*-
"""测试公共夹具：合成「像扫描古籍」的页面与 PDF。

为什么用合成图而不是真书？
    单元测试要的是**可控与可判定**：真书的纸面纹理、破洞分布无法预先断言。
    真书只用于 tests/acceptance 的终检（那里跑的是统计口径，不是精确断言）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pymupdf
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from classicsclean.config import DEFAULT_PARAMS, Params  # noqa: E402

SAMPLE_DIR = ROOT / "samples"


def make_scan_page(
    width: int = 640,
    height: int = 880,
    seed: int = 7,
    margin: int = 28,
    paper: float = 165.0,
    with_hole: bool = True,
    with_text: bool = True,
) -> np.ndarray:
    """生成一页仿真扫描件：黄纸底 + 霉斑 + 纸纹噪声 + 墨字 + 白边 + 破洞。

    返回 RGB uint8 数组。三个高斯霉斑是刻意做的**空间非均匀**黄变，
    正是「单一全局白点」失效的根因，也是局部白场存在的理由。
    """
    rng = np.random.default_rng(seed)
    g = np.full((height, width), float(paper), np.float32)
    yy, xx = np.mgrid[0:height, 0:width]

    # 低频黄斑 / 霉渍
    g -= 26.0 * np.exp(-(((yy - 0.35 * height) / 130.0) ** 2 + ((xx - 0.62 * width) / 160.0) ** 2))
    g -= 18.0 * np.exp(-(((yy - 0.78 * height) / 95.0) ** 2 + ((xx - 0.28 * width) / 120.0) ** 2))
    g -= 10.0 * np.exp(-(((yy - 0.55 * height) / 220.0) ** 2 + ((xx - 0.85 * width) / 90.0) ** 2))
    g += rng.normal(0.0, 2.2, g.shape)          # 纸纹噪声

    # 墨字：行列排布的短笔画
    if with_text:
        rows, cols = 14, 9
        x0 = margin + 60
        y0 = margin + 70
        step_x = (width - 2 * x0) // max(1, cols - 1)
        step_y = (height - 2 * y0) // max(1, rows - 1)
        canvas = np.zeros((height, width), np.float32)
        for r in range(rows):
            for c in range(cols):
                cx, cy = x0 + c * step_x, y0 + r * step_y
                cv2.rectangle(canvas, (cx - 9, cy - 12), (cx + 9, cy + 12), 1.0, -1)
                cv2.rectangle(canvas, (cx - 4, cy - 16), (cx + 4, cy + 16), 1.0, -1)
        strokes = cv2.GaussianBlur(canvas, (0, 0), 0.8)
        g = g * (1.0 - strokes * 0.78) + 52.0 * (strokes * 0.78)

    # 纯白扫描边
    g[:margin, :] = 252.0
    g[height - margin:, :] = 252.0
    g[:, :margin] = 252.0
    g[:, width - margin:] = 252.0

    # 破洞：图像内部不与白边连通的孤立亮块
    if with_hole:
        cv2.circle(g, (int(0.5 * width), int(0.15 * height)), 9, 248.0, -1)
        cv2.circle(g, (int(0.2 * width), int(0.62 * height)), 7, 246.0, -1)

    g = np.clip(g, 0, 255).astype(np.float32)
    # 纸张泛黄：R 偏高、B 偏低
    rgb = np.stack([np.clip(g + 14, 0, 255), np.clip(g + 2, 0, 255), np.clip(g - 16, 0, 255)], axis=2)
    return rgb.astype(np.uint8)


def write_scan_pdf(path, pages) -> str:
    """把若干 RGB 页面写成 PDF（无损 PNG 内嵌，保证测试输入不被压缩污染）。"""
    doc = pymupdf.open()
    for rgb in pages:
        h, w = rgb.shape[:2]
        page = doc.new_page(width=w, height=h)
        ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        assert ok
        page.insert_image(page.rect, stream=buf.tobytes())
    doc.save(str(path))
    doc.close()
    return str(path)


@pytest.fixture(scope="session")
def scan_page() -> np.ndarray:
    """单页合成扫描件。"""
    return make_scan_page()


@pytest.fixture(scope="session")
def scan_pdf(tmp_path_factory) -> str:
    """三页合成扫描 PDF。"""
    pages = [make_scan_page(seed=7), make_scan_page(seed=11), make_scan_page(seed=23)]
    path = tmp_path_factory.mktemp("pdf") / "sample_scan.pdf"
    return write_scan_pdf(path, pages)


@pytest.fixture(scope="session")
def real_pdf() -> str:
    """真书样本（缺失时跳过，不失败 —— 真书不是仓库的强制资产）。"""
    source = sorted(p for p in SAMPLE_DIR.glob("*.pdf") if "_清洁版" not in p.stem)
    if not source:
        pytest.skip(f"samples/ 下没有源 PDF（{SAMPLE_DIR}）")
    return str(source[0])


@pytest.fixture()
def params() -> Params:
    return DEFAULT_PARAMS
