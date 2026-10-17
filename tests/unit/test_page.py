# -*- coding: utf-8 -*-
"""单页流水线（level_page）端到端性质测试。"""
import cv2
import numpy as np
import pytest

from classicsclean.config import Params
from classicsclean.algorithms.masks import compute_masks
from classicsclean.pipeline.page import level_page


def test_output_basic_properties(scan_page, params):
    out, margin, info = level_page(scan_page, params)
    assert out.dtype == np.uint8
    assert out.shape == scan_page.shape[:2]
    assert out[margin].min() == 255, "白边必须恒为纯白"
    assert info["width"] == out.shape[1] and info["height"] == out.shape[0]
    assert 0.0 <= info["ink"] <= 1.0


def test_paper_becomes_white(scan_page, params):
    """纸面（排除白边/墨迹/破洞）应被拉到接近纯白。"""
    masks = compute_masks(scan_page, params)
    out, _, _ = level_page(scan_page, params, pre=masks)
    clean = masks.paper & ~(cv2.dilate(masks.ink.astype(np.uint8), np.ones((3, 3), np.uint8),
                                       iterations=3) > 0) & ~masks.holes
    assert float(np.median(out[clean])) >= 245.0, "纸面中位数未达到近白水平"
    assert float(np.median(masks.gray[clean])) < 200.0, "原始纸面本应明显偏暗（否则测不出增益）"


def _woodblock_page(size=(360, 360)):
    """模拟木版印刷：浓墨字芯外裹一层比纸面暗得不多、但明显更宽的洇墨晕边。

    全局墨迹阈值套不住晕边（晕边灰度高于 0.62×纸面中位），旧算法会把它当
    纸面漂白，导致笔画变细发灰——正是「文字偏浅」的成因。
    """
    g = np.full(size, 175.0, np.float32)
    g[:12, :] = g[-12:, :] = g[:, :12] = g[:, -12:] = 252.0     # 纯白扫描边
    # 一条竖直厚笔画：中心 x=180，芯宽 8，晕边再向两侧各 10px
    core = (slice(60, 300), slice(176, 184))
    halo = (slice(60, 300), slice(166, 194))
    g[halo] = 130.0
    g[core] = 30.0
    rgb = np.stack([np.clip(g + 10, 0, 255), g, np.clip(g - 10, 0, 255)], axis=2)
    return rgb.astype(np.uint8)


def test_faint_stroke_halo_is_preserved():
    """红线扩展：低对比洇墨晕边不得被 gamma 漂白，须按基线映射保留浓度。"""
    page = _woodblock_page()
    masks = compute_masks(page, Params())
    # 晕边外侧：距字芯 6-10px，超出默认文字保护的羽化范围（2px + 高斯羽化）
    left = (slice(60, 300), slice(166, 171))
    right = (slice(60, 300), slice(189, 194))

    washed, _, _ = level_page(page, Params(text_dev_thr=0), pre=masks)
    kept, _, _ = level_page(page, Params(), pre=masks)

    assert float(washed[left].mean()) > 220.0, "关闭扩展时晕边应被漂白（回归前提）"
    assert float(kept[left].mean()) < 200.0, f"晕边被漂白：左均值 {kept[left].mean():.1f}"
    assert float(kept[right].mean()) < 200.0, f"晕边被漂白：右均值 {kept[right].mean():.1f}"
    # 字芯必须仍然很黑，纸面必须仍然很白（不能顾此失彼）
    assert kept[60:300, 176:184].mean() < 60.0
    assert kept[12:60, 12:166].mean() > 240.0, "远纸面应保持近白"


def test_strokes_are_pixel_identical_to_baseline(scan_page, params):
    """红线：字芯输出必须逐像素等于「基线白场 + 线性映射」，差值为 0。"""
    baseline_params = Params(gamma=1.0, wp_mode="whitest", text_guard=1)
    masks = compute_masks(scan_page, params)
    new, _, _ = level_page(scan_page, params, pre=masks)
    base, _, _ = level_page(scan_page, baseline_params, pre=masks)
    core = cv2.erode(masks.ink.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=2) > 0
    assert core.any()
    diff = int(np.abs(new[core].astype(np.int16) - base[core].astype(np.int16)).max())
    assert diff == 0, f"字芯被改动：最大差 {diff}"


def _dark_tinted_page(size=(400, 400)):
    """模拟深红封面：纸面整体偏暗，但存在一块明显更亮（却非白边）的区域。

    此时「最白区白点」远高于纸面中位数，纸面亮度会跌破 200 —— 必须触发兜底。
    """
    g = np.full(size, 120.0, np.float32)
    g[120:280, 120:280] = 200.0                      # 明亮但位于图像内部的块
    g[60:70, 80:320] = 45.0                          # 若干笔画
    g[300:310, 80:320] = 45.0
    g[:8, :] = g[-8:, :] = g[:, :8] = g[:, -8:] = 252.0   # 纯白扫描边
    rgb = np.stack([np.clip(g + 40, 0, 255), g, g], axis=2)   # 强红偏
    return rgb.astype(np.uint8)


def test_safeguard_engaged_on_dark_page():
    """强色偏/低对比页（纸面亮度 < 200）必须触发兜底白场，背景仍读作白。"""
    params = Params(wp_mode="whitest", gamma=1.0, text_guard=1)
    out, margin, info = level_page(_dark_tinted_page(), params)
    assert info["paper_level"] < 200.0, "构造的页面本应触发兜底阈值"
    assert info["mode"] == "paper-level(safeguard)"
    assert out[margin].min() == 255
    masks = compute_masks(_dark_tinted_page(), params)
    paper_out = float(np.median(out[masks.paper]))
    assert paper_out > 200.0, f"兜底后纸面仍应偏白，实测 {paper_out}"


def test_gamma_vs_linear_background(scan_page):
    """同等条件下，gamma 曲线的纸面一定比线性映射更白。"""
    masks = compute_masks(scan_page, Params())
    lin, _, _ = level_page(scan_page, Params(gamma=1.0), pre=masks)
    gam, _, _ = level_page(scan_page, Params(gamma=3.0), pre=masks)
    clean = masks.paper & ~masks.holes
    assert float(np.median(gam[clean])) > float(np.median(lin[clean]))


@pytest.mark.parametrize("wp_mode", ["local", "whitest", "p65"])
def test_all_white_point_modes_run(scan_page, wp_mode):
    """所有白场模式都必须能跑通并产出合法图像。"""
    out, _, info = level_page(scan_page, Params(wp_mode=wp_mode))
    assert out.dtype == np.uint8 and out.size > 0
    assert np.isfinite(info["Wp"])
