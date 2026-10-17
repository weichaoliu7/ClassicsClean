# -*- coding: utf-8 -*-
"""掩膜分割单元测试：白边 / 墨迹 / 破洞 / 纸面 必须互斥且各司其职。"""
import cv2
import numpy as np

from classicsclean.config import Params
from classicsclean.algorithms.masks import compute_masks, hole_mask, white_margin_mask

MARGIN_PX = 28


def test_margin_only_connected_to_border(scan_page, params):
    """白边掩膜只能覆盖与图像边缘连通的亮区，不包含内部亮块。"""
    masks = compute_masks(scan_page, params)
    h, w = masks.margin.shape
    assert masks.margin[:MARGIN_PX, :].all(), "上边缘必须整条判为白边"
    assert masks.margin[-MARGIN_PX:, :].all(), "下边缘必须整条判为白边"
    assert masks.margin[:, :MARGIN_PX].all(), "左边缘必须整条判为白边"
    assert masks.margin[:, -MARGIN_PX:].all(), "右边缘必须整条判为白边"
    # 图像中心区域不应是白边
    assert not masks.margin[h // 2 - 20:h // 2 + 20, w // 2 - 20:w // 2 + 20].any()


def test_hole_is_not_margin(scan_page, params):
    """破洞（内部孤立亮块）必须与白边互斥。"""
    masks = compute_masks(scan_page, params)
    assert masks.holes.any(), "合成页面应能检出破洞"
    assert not np.any(masks.holes & masks.margin), "破洞不得与白边重叠"


def test_masks_are_exclusive(scan_page, params):
    """四类掩膜两两互斥，纸面 = 非白边 & 非墨迹 & 非破洞。"""
    gray, margin, ink, holes, paper = compute_masks(scan_page, params)
    assert not np.any(margin & ink)
    assert not np.any(margin & holes)
    assert not np.any(ink & holes)
    assert not np.any(paper & (margin | ink | holes))
    expected = (~margin) & (~ink) & (~holes)
    assert np.array_equal(paper, expected)


def test_ink_covers_strokes(scan_page, params):
    """墨迹掩膜应覆盖大部分笔画像素（合成页的笔画灰度约 52）。"""
    masks = compute_masks(scan_page, params)
    dark = masks.gray < 90
    recall = float((masks.ink & dark).sum() / max(1, dark.sum()))
    assert recall > 0.9, f"墨迹召回率过低：{recall:.3f}"


def test_hole_mask_area_bounds():
    """破洞判定遵守面积上下限。"""
    gray = np.full((200, 200), 120.0, np.float32)
    gray[50:60, 50:60] = 250.0        # 100 px 亮块 -> 应判为破洞
    gray[150:152, 150:152] = 250.0    # 4 px 亮点 -> 面积过小，不判
    holes = hole_mask(gray, np.zeros((200, 200), bool), bright_thr=225.0,
                      min_area=30, max_frac=0.03)
    assert holes[55, 55], "100px 的亮块应判为破洞"
    assert not holes[151, 151], "4px 的亮点不应判为破洞"


def test_margin_fallback_when_no_seed():
    """没有边缘种子（整页无亮边）时退化为「全亮区」，不应抛异常。"""
    gray = np.full((100, 100), 200.0, np.float32)
    gray[10:20, 10:20] = 250.0
    margin = white_margin_mask(gray, thr=235.0)
    assert margin.dtype == bool
    assert margin[15, 15], "无种子时亮区整体视为白边"
