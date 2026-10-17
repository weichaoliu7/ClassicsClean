# -*- coding: utf-8 -*-
"""【第 2 层 · 算法】破洞修补。

输入：输出图 + 灰度图 + 白边 / 墨迹 / 破洞掩膜
输出：(修补后的图, 已补洞数, 因碰字跳过数)
边界：只补**完全不涉及文字**的洞；碰字的洞原样放过（红线 4）

做法：把洞及其暗边用周边纸面色调羽化填充。
判据：洞外围 8px 内若存在延伸到区域之外的墨迹分量，说明该墨迹是笔画而非洞边，
直接跳过（宁可不补，也绝不动字）。
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

from ..morphology import (
    bbox_of,
    closing,
    connected_components,
    dilate,
    disk,
    gaussian_blur,
    gaussian_radius,
)

__all__ = ["repair_holes"]

GAUSSAIN_SIGMA = 3.0
"""破洞修补羽化的高斯 sigma（与历史版本一致，勿随调参改动）。"""

CLOSE_RADIUS = 2
"""闭运算结构元半径（disk(2)，即 5x5 核），用于封住破洞内的小断裂。"""


def repair_holes(
    out: np.ndarray,
    gray: np.ndarray,
    margin: np.ndarray,
    ink: np.ndarray,
    holes: np.ndarray,
    rim: int = 8,
    fill_extra: int = 3,
) -> Tuple[np.ndarray, int, int]:
    """返回 (修补后的图像, 已补洞数, 因碰字跳过数)。

    性能：每个破洞的形态学 / 高斯 / 混合全部限制在「破洞包围盒 ± crop_radius」
    内进行。这些算子都是局部算子，盒外结果恒为 0 / False，与整页计算逐像素
    一致；破洞密集的页面（数百个洞）因此从分钟级降到秒级。裁剪半径按各算子的
    结构元半径累加：闭运算 CLOSE_RADIUS + 填充外扩 fill_extra + 环 ring_radius。
    """
    if not holes.any():
        return out, 0, 0

    ink_lab = connected_components(ink, connectivity=8).labels
    hole_comp = connected_components(holes, connectivity=8)
    ink_area = np.bincount(ink_lab.ravel(), minlength=int(ink_lab.max()) + 1)
    h, w = gray.shape
    ring_radius = rim + 6                    # ring = dilate(fill_mask, disk(rim+6))
    crop_radius = CLOSE_RADIUS + fill_extra + ring_radius
    sigma_r = gaussian_radius(GAUSSAIN_SIGMA)

    filled = skipped = 0
    res = out.astype(np.float32)
    for i in range(1, hole_comp.count + 1):
        box = hole_comp.slices[i]
        r0, r1 = box[0].start, box[0].stop
        c0, c1 = box[1].start, box[1].stop
        # 裁剪盒：破洞包围盒外扩 crop_radius（clamp 到图内）
        R0, R1 = max(r0 - crop_radius, 0), min(r1 + crop_radius, h)
        C0, C1 = max(c0 - crop_radius, 0), min(c1 + crop_radius, w)

        comp = hole_comp.labels[R0:R1, C0:C1] == i
        region = dilate(comp, disk(rim))

        touches_text = False
        ink_sub = ink[R0:R1, C0:C1]
        ink_lab_sub = ink_lab[R0:R1, C0:C1]
        for cid in np.unique(ink_lab_sub[region & ink_sub]):
            if cid == 0:
                continue
            total = int(ink_area[cid])
            inside = int(((ink_lab_sub == cid) & region).sum())
            if total == 0:
                continue
            if inside / total < 0.85:  # 笔画延伸到区域之外 -> 是字，不是洞边
                touches_text = True
                break
        if touches_text:
            skipped += 1
            continue

        margin_sub = margin[R0:R1, C0:C1]
        fill_mask = closing(comp, disk(CLOSE_RADIUS))
        fill_mask = dilate(fill_mask, disk(fill_extra))
        fill_mask &= ~margin_sub

        ring = dilate(fill_mask, disk(ring_radius))
        ring &= ~fill_mask
        ring &= ~ink_sub
        ring &= ~margin_sub
        if ring.sum() < 50:
            continue

        res_sub = res[R0:R1, C0:C1]
        fill_val = float(np.median(res_sub[ring]))

        # 高斯只需算到 fill_mask 包围盒外扩一个核半径：之外 fill_mask 全 0，
        # reflect 边界读数与整页计算相同，所以 alpha 逐像素一致。
        # 注意外扩量是 sigma_r + 1 而不是 sigma_r：np.pad 的 reflect 不重复边缘
        # 元素，左补区镜像的是作物的第 1..sigma_r 列（而非第 0 列），作物必须
        # 再往外让一列，才能保证镜像进来的全是 0。
        alpha = np.zeros(fill_mask.shape, np.float32)
        fbox = bbox_of(fill_mask)
        if fbox is not None:
            fr0, fr1, fc0, fc1 = fbox
            sr0, sr1 = max(fr0 - sigma_r - 1, 0), min(fr1 + sigma_r + 1, fill_mask.shape[0])
            sc0, sc1 = max(fc0 - sigma_r - 1, 0), min(fc1 + sigma_r + 1, fill_mask.shape[1])
            alpha[sr0:sr1, sc0:sc1] = gaussian_blur(
                fill_mask[sr0:sr1, sc0:sc1].astype(np.float32), GAUSSAIN_SIGMA
            )
        alpha = np.clip(alpha, 0.0, 1.0)
        # 破洞内的墨迹像素不得被填充色覆盖：它们是有内容的笔画（红线「字不动」），
        # 只有纸面破损部分才用周边纸面色修补。否则洞内的字会被抹掉，
        # 而且填充色会随着周边晕边微调而改变，破坏字芯逐像素等于基线的红线。
        alpha[ink_sub] = 0.0
        res[R0:R1, C0:C1] = res_sub * (1.0 - alpha) + fill_val * alpha
        filled += 1

    res[margin] = 255.0
    return np.clip(res, 0, 255).astype(np.uint8), filled, skipped
