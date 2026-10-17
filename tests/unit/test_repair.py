# -*- coding: utf-8 -*-
"""破洞修补单元测试：不碰字的洞要补，碰字的洞必须原样放过。

另有一组「裁剪等价性」测试：repair_holes 为性能把每个破洞限制在包围盒内
计算，结果必须与朴素的整页实现逐像素一致（含贴边破洞、碰字破洞、密集破洞）。
"""
import cv2
import numpy as np
import pytest

from classicsclean.algorithms.repair import repair_holes
from classicsclean.morphology import closing, connected_components, dilate, disk, gaussian_blur


def _page_with_hole(hole_center, size=(300, 300), ink_rect=None):
    gray = np.full(size, 170.0, np.float32)
    margin = np.zeros(size, bool)
    margin[:10, :] = margin[-10:, :] = margin[:, :10] = margin[:, -10:] = True
    gray[margin] = 252.0

    ink = np.zeros(size, bool)
    if ink_rect is not None:
        (x0, y0, x1, y1) = ink_rect
        ink[y0:y1, x0:x1] = True
        gray[ink] = 52.0

    holes = np.zeros(size, bool)
    cy, cx = hole_center
    cv2.circle(holes, (cx, cy), 12, 1, -1)   # 比 min_area=30 大得多
    gray[holes] = 250.0
    return gray, margin, ink, holes


def test_hole_away_from_text_is_filled():
    """不涉及文字的破洞应被周边纸面色填充（亮度回落到纸面水平）。"""
    gray, margin, ink, holes = _page_with_hole((150, 150))
    out, filled, skipped = repair_holes(gray.copy(), gray, margin, ink, holes)
    assert filled == 1 and skipped == 0
    assert out[holes].mean() < gray[holes].mean() - 40, "洞内应被纸面色覆盖，明显变暗"


def test_hole_touching_text_is_skipped():
    """与笔画相交的洞必须跳过 —— 宁可不补，绝不动字。"""
    gray, margin, ink, holes = _page_with_hole((150, 150), ink_rect=(120, 60, 180, 240))
    out, filled, skipped = repair_holes(gray.copy(), gray, margin, ink, holes)
    assert skipped == 1 and filled == 0
    assert np.allclose(out[ink], gray[ink]), "笔画像素不得被改动"


def test_no_holes_is_noop():
    gray, margin, ink, _ = _page_with_hole((150, 150))
    holes = np.zeros_like(margin)
    out, filled, skipped = repair_holes(gray.copy(), gray, margin, ink, holes)
    assert filled == 0 and skipped == 0
    assert np.array_equal(out, gray)


def test_margin_stays_white_after_repair():
    gray, margin, ink, holes = _page_with_hole((150, 40))
    out, _, _ = repair_holes(gray.copy(), gray, margin, ink, holes)
    assert out[margin].min() == 255


# ---------------------------------------------------------------- 裁剪等价性
def _repair_reference_full_page(out, gray, margin, ink, holes, rim=8, fill_extra=3):
    """朴素整页参考实现：与 repair_holes 优化前的第一版逐行等价。

    存在的意义就是当 repair_holes 的「金丝雀」——任何裁剪bug都会在这里现形。
    """
    if not holes.any():
        return out, 0, 0
    ink_lab = connected_components(ink, connectivity=8).labels
    hole_comp = connected_components(holes, connectivity=8)
    filled = skipped = 0
    res = out.astype(np.float32)
    for i in range(1, hole_comp.count + 1):
        comp = hole_comp.labels == i
        region = dilate(comp, disk(rim))
        touches = False
        for cid in np.unique(ink_lab[region & ink]):
            if cid == 0:
                continue
            total = int((ink_lab == cid).sum())
            inside = int(((ink_lab == cid) & region).sum())
            if total and inside / total < 0.85:
                touches = True
                break
        if touches:
            skipped += 1
            continue
        fm = closing(comp, disk(2))
        fm = dilate(fm, disk(fill_extra))
        fm &= ~margin
        ring = dilate(fm, disk(rim + 6))
        ring &= ~fm
        ring &= ~ink
        ring &= ~margin
        if ring.sum() < 50:
            continue
        fill_val = float(np.median(res[ring]))
        alpha = np.clip(gaussian_blur(fm.astype(np.float32), 3.0), 0.0, 1.0)
        alpha[ink] = 0.0
        res = res * (1.0 - alpha) + fill_val * alpha
        filled += 1
    res[margin] = 255.0
    return np.clip(res, 0, 255).astype(np.uint8), filled, skipped


def _busy_page(seed=5, size=(260, 340)):
    """一页「很忙」的合成件：多个破洞（含贴边、碰字、微小环），纸面带噪声。"""
    rng = np.random.default_rng(seed)
    h, w = size
    gray = (170.0 + rng.normal(0, 3, (h, w))).astype(np.float32)
    margin = np.zeros((h, w), bool)
    margin[:12, :] = margin[-12:, :] = margin[:, :12] = margin[:, -12:] = True
    gray[margin] = 252.0
    yy, xx = np.mgrid[0:h, 0:w]
    holes = np.zeros((h, w), bool)
    for cy, cx, r in ((60, 80, 9), (60, 250, 6), (150, 60, 7), (150, 300, 5),
                      (240, 120, 8), (240, 220, 4), (5, 170, 6), (130, 170, 11)):
        holes |= ((yy - cy) ** 2 + (xx - cx) ** 2) <= r * r
    gray[holes] = 250.0
    ink = np.zeros((h, w), bool)
    ink[55:75, 70:100] = True        # 第一个破洞旁边的“字”
    ink[140:165, 290:310] = True     # 第四个破洞旁边的“字”
    gray[ink] = 52.0
    out = np.clip(gray * 0.9 + 20, 0, 255).astype(np.uint8)
    return out, gray, margin, ink, holes


@pytest.mark.parametrize("seed", (5, 11, 23))
def test_cropped_repair_matches_full_page(seed):
    """包围盒裁剪优化必须与整页实现逐像素一致（含 filled/skipped 计数）。"""
    out, gray, margin, ink, holes = _busy_page(seed)
    fast = repair_holes(out.copy(), gray, margin, ink, holes)
    ref = _repair_reference_full_page(out.copy(), gray, margin, ink, holes)
    assert (fast[1], fast[2]) == (ref[1], ref[2]), "补洞/跳过计数不一致"
    assert np.array_equal(fast[0], ref[0]), "裁剪版与整页版输出不一致"
