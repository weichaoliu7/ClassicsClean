# -*- coding: utf-8 -*-
"""【第 2 层 · 算法】白场（白点）估计。

输入：PageMasks + 基线白点 Wp + 纸面亮度
输出：(Wp_map, Wp_full, mode)
        Wp_map  —— 逐像素白场，参与最终亮度映射
        Wp_full —— 基线白场，仅供文字保护区使用（字必须沿用基线映射）
边界：只在纸面上统计；与白边 / 墨迹 / 破洞的重叠像素恒为 0

三类白场：
  1. whitest —— 纸面「最白连通区域」的均值（面积自适应且不小于整页 0.8%）。
     这是全书的基线白点 Wp，同时也是文字保护区使用的固定基线 Wp_full。
  2. local   —— 逐点局部纸面亮度场 bg_field()：每个位置用它自己附近的纸面亮度
     做白点。线装纸的黄变/霉斑是空间非均匀的，单一全局白点在数学上无法同时把
     亮区与暗区都推到白，因此定稿改用局部白场。
  3. 兜底    —— 纸面亮度低于 200（深红封面等强色偏页）时，白场退化为
     「局部均值 + 0.8×局部标准差」，保证背景仍读作白色。
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from ..config import PERCENTILE_MODES, Params
from ..morphology import box_blur, connected_components, max_filter, median_blur
from .masks import PageMasks

__all__ = [
    "growth_ladder",
    "whitest_region",
    "bg_field",
    "safeguard_field",
    "local_bg_stat",
    "local_envelope",
    "resolve_white_field",
]


def growth_ladder(min_frac: float, steps: int = 24) -> Tuple[float, ...]:
    """白区候选集的增长阶梯。

    必须从 min_frac 起步，否则调小面积下限时不会生效。
    """
    ladder = list(np.geomspace(max(min_frac, 1e-6), 1.0, steps))
    ladder[0] = min_frac
    ladder.append(1.0)
    return tuple(sorted(set(round(float(v), 8) for v in ladder)))


def whitest_region(
    gray: np.ndarray,
    paper: np.ndarray,
    min_frac: float,
    outlier_k: float = 2.5,
    return_mask: bool = False,
):
    """在纸面内自适应的找出「最白连通区域」，返回 (均值, 面积占比[, 区域掩膜])。

    逐步扩大亮度候选集，直到最大连通分量的面积 >= min_frac × 整页面积；
    同时要求该区域均值 <= 纸面均值 + outlier_k × 纸面标准差，
    否则判定为破洞/高光噪点并继续扩大（门槛调小后极易选中亮点，此约束必不可少）。
    """
    n_total = gray.size
    pg = gray[paper]
    n_paper = pg.size
    if n_paper == 0:
        empty = np.zeros(gray.shape, bool)
        return (float(gray.max()), 0.0, empty) if return_mask else (float(gray.max()), 0.0)

    order = np.sort(pg)[::-1]
    k0 = min(n_paper, max(1, int(np.ceil(min_frac * n_total))))
    fb_mask = paper & (gray >= order[k0 - 1])
    fallback = float(order[:k0].mean())
    pmean = float(np.mean(pg))
    pstd = float(np.std(pg))
    cap = pmean + outlier_k * pstd

    chosen = None
    for frac in growth_ladder(min_frac):
        k = min(n_paper, max(1, int(np.ceil(frac * n_total))))
        thr = order[k - 1]
        cand = paper & (gray >= thr)
        comp = connected_components(cand, connectivity=8)
        if comp.count > 0:
            i = 1 + int(np.argmax(comp.areas[1:]))
            if comp.areas[i] >= min_frac * n_total:
                sel = comp.labels == i
                val = float(gray[sel].mean())
                area = float(comp.areas[i] / n_total)
                if chosen is None:
                    chosen = (val, area, sel)
                if val <= cap:
                    return (val, area, sel) if return_mask else (val, area)
    if chosen is not None:
        val, area, sel = chosen
        return (val, area, sel) if return_mask else (val, area)
    return (fallback, 0.0, fb_mask) if return_mask else (fallback, 0.0)


def bg_field(gray: np.ndarray, strict: np.ndarray, wins=(31, 61, 121)) -> np.ndarray:
    """多尺度局部纸面亮度场。

    只在「确定是纸面」的区域上统计，由粗到细逐级细化：窗口内纸面样本不足时
    回退到上一级更粗的估计，保证字密集处也有可靠的局部纸面亮度。
    """
    env: Optional[np.ndarray] = None
    pm = strict.astype(np.float32)
    for w in wins:
        num = box_blur(gray * pm, w)
        den = box_blur(pm, w)
        cur = num / np.maximum(den, 1e-6)
        env = cur if env is None else np.where(den > 0.25, cur, env)
    return env


def safeguard_field(
    gray: np.ndarray,
    paper: np.ndarray,
    paper_med: float,
    win: int = 61,
    k: float = 0.8,
) -> np.ndarray:
    """兜底白场：局部纸面均值 + k × 局部标准差（强色偏/低对比页面使用）。"""
    pmf = paper.astype(np.float32)
    den0 = box_blur(pmf, win)
    num = box_blur(gray * pmf, win)
    den = np.maximum(den0, 1e-6)
    cov = den0 > 0.15
    lmean = np.where(cov, num / den, paper_med)
    sqn = box_blur((gray * gray) * pmf, win)
    lstd = np.sqrt(np.maximum(sqn / den - lmean ** 2, 0.0))
    default_std = max(float(np.std(gray[paper])) if paper.any() else 5.0, 3.0)
    lstd = np.where(cov, lstd, default_std)
    return np.clip(lmean + k * lstd, 1.0, 255.0)


def local_bg_stat(gray: np.ndarray, paper: np.ndarray, win: int = 31, k: float = 1.0) -> np.ndarray:
    """局部纸面「上尾」估计 = 局部均值 + k × 局部标准差（在纸面掩膜上统计）。

    比局部最大值稳健：不受孤立亮点污染，且能自适应纸面纹理强度。
    """
    pm = paper.astype(np.float32)
    den0 = box_blur(pm, win)
    den = np.maximum(den0, 1e-6)
    lmean = box_blur(gray * pm, win) / den
    sqn = box_blur((gray * gray) * pm, win)
    lstd = np.sqrt(np.maximum(sqn / den - lmean ** 2, 0.0))
    return lmean + k * lstd


def local_envelope(gray: np.ndarray, paper: np.ndarray, iters: int = 32, med: int = 5) -> np.ndarray:
    """局部纸面上包络：把纸面亮度沿邻域取最大值扩散 iters 次。"""
    g = np.where(paper, np.clip(gray, 0, 255), 0.0).astype(np.uint8)
    if med and med > 1:
        g = median_blur(g, med)
    return max_filter(g, int(iters)).astype(np.float32)


def resolve_white_field(
    masks: PageMasks,
    wp: float,
    paper_med: float,
    paper_level: float,
    params: Params,
):
    """决定本页实际使用的白场图。

    返回 (Wp_map, Wp_full, mode)：
      Wp_map  —— 逐像素白场，参与最终亮度映射；
      Wp_full —— 基线白场，仅供文字保护区使用（文字必须逐像素沿用基线映射）；
      mode    —— 白场模式说明，写进处理日志与报告。
    """
    gray, margin, ink, holes, paper = masks
    wp_map = np.full_like(gray, float(wp))
    wp_full = wp_map
    mode = "whitest-region"

    if params.wp_mode in PERCENTILE_MODES:
        mode = f"paper-p{int(params.wp_mode[1:])}"
    if params.wp_mode == "local":
        wp_map = bg_field(gray, masks.strict_paper, params.bg_wins)
        mode = "local-paper-field"

    if paper_level < params.safeguard_level:
        wp_map = safeguard_field(gray, paper, paper_med, params.safeguard_win, params.safeguard_k)
        wp_full = wp_map  # 兜底页没有独立基线，字与背景需共用同一白场
        if mode != "local-paper-field":
            mode = "paper-level(safeguard)"

    return wp_map, wp_full, mode
