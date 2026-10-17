# -*- coding: utf-8 -*-
"""色调映射与文字保护单元测试。

最关键的断言是 `test_text_guard_keeps_strokes_identical`：
在保护区权重为 1 的位置，输出必须**逐像素**等于基线线性映射的结果。
"""
import cv2
import numpy as np

from classicsclean.config import Params
from classicsclean.algorithms.masks import compute_masks
from classicsclean.algorithms.tone import (
    apply_text_guard,
    soft_text_weight,
    text_guard_weight,
    tone_map,
)


def _flat_page(shape=(200, 200)):
    gray = np.full(shape, 170.0, np.float32)
    margin = np.zeros(shape, bool)
    return gray, margin


def test_tone_map_margin_is_pure_white():
    """白边恒为 255，任何曲线都不允许改动。"""
    gray, margin = _flat_page()
    margin[:10, :] = True
    for curve in [("linear",), ("gamma", 3.0), ("knee", 0.3, 12.0)]:
        out = tone_map(gray, margin, np.full_like(gray, 180.0), 60.0, curve)
        assert out[margin].min() == 255 and out[margin].max() == 255


def test_tone_map_monotonic():
    """输入越亮输出越亮，映射必须单调（且亮于白点的部分一律压到 255）。"""
    gray, margin = _flat_page((1, 256))
    gray = np.arange(256, dtype=np.float32).reshape(1, 256)
    out = tone_map(gray, margin, np.full_like(gray, 200.0), 40.0, ("gamma", 3.0))
    assert np.all(np.diff(out[0]) >= 0), "输出必须随输入灰度单调不减"
    assert out[0, 200:].min() == 255, "不暗于白点的像素必须映射为纯白"


def test_gamma_pushes_background_whiter_than_linear():
    """高光压缩的作用：把纸面残留底灰推向白（这是 v13 的核心增益）。"""
    gray, margin = _flat_page((50, 50))
    wp, bp = np.full_like(gray, 180.0), 40.0
    linear = tone_map(gray, margin, wp, bp, ("linear",))
    gamma = tone_map(gray, margin, wp, bp, ("gamma", 3.0))
    assert gamma.mean() > linear.mean(), "gamma 曲线必须让背景更白"


def test_unknown_curve_rejected():
    gray, margin = _flat_page()
    try:
        tone_map(gray, margin, np.full_like(gray, 180.0), 40.0, ("bogus",))
    except ValueError:
        return
    raise AssertionError("未知曲线应抛 ValueError")


def test_text_guard_weight_inner_is_exactly_one(scan_page):
    """内圈权重必须严格为 1 —— 羽化内圈会让字被改动（历史教训）。"""
    masks = compute_masks(scan_page, Params())
    w = text_guard_weight(masks.ink, radius=1, sigma=1.5)
    assert w[masks.ink].min() == 1.0, "笔画上的权重必须恒为 1"


def test_text_guard_keeps_strokes_identical(scan_page, params):
    """红线验证：字芯输出 == 基线线性映射输出，差异必须为 0。"""
    masks = compute_masks(scan_page, params)
    w = text_guard_weight(masks.ink, int(params.text_guard), params.text_guard_sigma)
    baseline = tone_map(masks.gray, masks.margin, np.full_like(masks.gray, 180.0), 40.0,
                        ("linear",))
    mapped = tone_map(masks.gray, masks.margin, np.full_like(masks.gray, 190.0), 40.0,
                      ("gamma", 3.0))
    blended = apply_text_guard(mapped, baseline, w)
    core = cv2.erode(masks.ink.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=2) > 0
    assert core.any()
    diff = np.abs(blended[core].astype(np.int16) - baseline[core].astype(np.int16))
    assert int(diff.max()) == 0, f"字芯被改动了，最大差 {int(diff.max())}"


def test_apply_text_guard_full_weight_equals_baseline():
    gray = np.full((10, 10), 100, np.uint8)
    base = np.full((10, 10), 200, np.uint8)
    assert np.array_equal(apply_text_guard(gray, base, np.ones((10, 10), np.float32)), base)


def test_soft_text_weight_range(scan_page):
    masks = compute_masks(scan_page, Params())
    w = soft_text_weight(masks.gray, masks, threshold=15.0)
    assert w.min() >= 0.0 and w.max() <= 1.0
    assert w[masks.ink].mean() > 0.5, "笔画处的软权重应接近 1"
