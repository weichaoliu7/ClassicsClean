# -*- coding: utf-8 -*-
"""白场估计单元测试。核心不变式：白区只能落在纸面上，绝不碰白边 / 墨迹 / 破洞。"""
import numpy as np

from classicsclean.config import Params
from classicsclean.algorithms.masks import compute_masks
from classicsclean.algorithms.white_point import (
    bg_field,
    growth_ladder,
    safeguard_field,
    whitest_region,
)


def test_whitest_region_never_touches_excluded_areas(scan_page, params):
    """最白区域与白边、墨迹、破洞的重叠像素必须严格为 0。"""
    masks = compute_masks(scan_page, params)
    _, area, region = whitest_region(masks.gray, masks.paper, params.min_area_frac,
                                     params.outlier_k, return_mask=True)
    assert int((region & masks.margin).sum()) == 0, "白区不得落在纯白边上"
    assert int((region & masks.ink).sum()) == 0, "白区不得落在墨迹上"
    assert int((region & masks.holes).sum()) == 0, "白区不得落在破洞上"
    assert area >= params.min_area_frac, f"白区面积应达标：{area:.4%}"


def test_whitest_region_value_is_paper_tone(scan_page, params):
    """白点必须是纸面色调量级，不能被孤立亮点带偏（outlier 约束生效）。"""
    masks = compute_masks(scan_page, params)
    wp, _ = whitest_region(masks.gray, masks.paper, params.min_area_frac, params.outlier_k)
    paper_mean = float(masks.gray[masks.paper].mean())
    paper_std = float(masks.gray[masks.paper].std())
    assert wp <= paper_mean + params.outlier_k * paper_std + 1e-6, "白点超过纸面上尾，疑似选中破洞"
    assert wp >= float(np.median(masks.gray[masks.paper])), "白点不应低于纸面中位数"


def test_growth_ladder_starts_from_min():
    """增长阶梯必须从面积下限起步，否则调小下限不会生效。"""
    ladder = growth_ladder(0.008)
    assert abs(ladder[0] - 0.008) < 1e-9
    assert ladder[-1] == 1.0
    assert all(ladder[i] <= ladder[i + 1] for i in range(len(ladder) - 1))


def test_bg_field_tracks_paper_brightness(scan_page, params):
    """局部白场应贴近纸面亮度，且处处有限。"""
    masks = compute_masks(scan_page, params)
    field = bg_field(masks.gray, masks.strict_paper, params.bg_wins)
    assert field.shape == masks.gray.shape
    assert np.isfinite(field).all()
    lo = float(np.percentile(masks.gray[masks.paper], 1))
    hi = float(np.percentile(masks.gray[masks.paper], 99))
    assert field[masks.paper].min() >= lo - 25, "局部白场不应显著低于纸面"
    assert field[masks.paper].max() <= hi + 25, "局部白场不应显著高于纸面"


def test_safeguard_field_bounded(scan_page, params):
    """兜底白场取值必须落在合法灰度区间内。"""
    masks = compute_masks(scan_page, params)
    field = safeguard_field(masks.gray, masks.paper, float(np.median(masks.gray[masks.paper])),
                            params.safeguard_win, params.safeguard_k)
    assert field.min() >= 1.0 and field.max() <= 255.0
