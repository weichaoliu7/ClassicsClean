# -*- coding: utf-8 -*-
"""【第 2 层 · 算法】页面掩膜分割：纯白边 / 墨迹 / 破洞 / 纸面。

输入：RGB uint8 位图 + Params
输出：PageMasks —— gray / margin / ink / holes / paper，后四者**互斥且覆盖全图**
边界：不做任何亮度变换，只回答「这一片像素是什么」

后续所有统计（白点、黑场、洁净度）都只在「纸面」上进行，
从而保证指标不被白边、墨字、破洞污染。
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np

from ..config import Params
from ..imaging import to_gray
from ..morphology import connected_components, dilate, region_from_seeds, square

__all__ = ["PageMasks", "compute_masks", "white_margin_mask", "hole_mask"]


class PageMasks(NamedTuple):
    """单页掩膜集合（全部为与图像同尺寸的 bool 数组）。"""

    gray: np.ndarray      # float32 灰度
    margin: np.ndarray    # 纯白扫描边（与图像边缘连通的亮区）
    ink: np.ndarray       # 墨迹（笔画）
    holes: np.ndarray     # 破洞（图像内部孤立亮块）
    paper: np.ndarray     # 纸面 = 非白边 & 非墨迹 & 非破洞

    @property
    def strict_paper(self) -> np.ndarray:
        """严格纸面：纸面再排除墨迹膨胀 2px 的边缘区，用于局部白场统计。"""
        dil = dilate(self.ink, square(2))
        strict = self.paper & ~dil
        return strict if strict.any() else self.paper


def white_margin_mask(gray: np.ndarray, thr: float = 235.0) -> np.ndarray:
    """与图像边缘连通的亮区 = 扫描仪的纯白边距。

    边缘种子点做 4 连通漫水扩散；完全没有种子时退化为「所有亮区」，
    这类页面（整页发白的淡墨页）输出会被强制刷白，这是预期行为。
    """
    bright = gray >= float(thr)
    seed = np.zeros(bright.shape, dtype=bool)
    seed[0, :] = seed[-1, :] = seed[:, 0] = seed[:, -1] = True
    seed &= bright
    if not seed.any():
        return bright          # 无边缘种子：所有亮区整体视为白边
    return region_from_seeds(bright, seed, connectivity=4)


def hole_mask(
    gray: np.ndarray,
    margin: np.ndarray,
    bright_thr: float = 225.0,
    min_area: int = 30,
    max_frac: float = 0.03,
) -> np.ndarray:
    """图像内部（不与白边连通）的孤立亮块 = 纸张破损/虫蛀孔。"""
    n_total = gray.size
    cand = (gray >= float(bright_thr)) & (~margin)
    comp = connected_components(cand, connectivity=8)
    holes = np.zeros(gray.shape, bool)
    for i in range(1, comp.count + 1):
        if min_area <= int(comp.areas[i]) <= max_frac * n_total:
            holes |= comp.labels == i
    return holes


def compute_masks(rgb: np.ndarray, params: Params) -> PageMasks:
    """RGB 位图 -> 掩膜集合。"""
    gray = to_gray(rgb)
    margin = white_margin_mask(gray, params.margin_thr)
    non_margin = gray[~margin]
    base = float(np.median(non_margin)) if non_margin.size else 200.0
    ink = (~margin) & (gray < params.ink_ratio * base)
    holes = hole_mask(gray, margin, params.hole_thr, params.hole_min_area, params.hole_max_frac)
    paper = (~margin) & (~ink) & (~holes)
    return PageMasks(gray=gray, margin=margin, ink=ink, holes=holes, paper=paper)
