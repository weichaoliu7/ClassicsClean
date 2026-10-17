# -*- coding: utf-8 -*-
"""【第 3 层 · 编排】单页流水线：把掩膜、白场、色调映射、破洞修补串起来。

输入：RGB uint8 位图 + Params（可选传入预计算掩膜 pre，用于同页多方案对比）
输出：(uint8 灰度图, 白边掩膜, 统计信息 dict)
边界：不读文件、不写文件、不接触界面

数据流：
    RGB 位图 -> compute_masks -> 基线白点 Wp -> 黑场 bp
             -> resolve_white_field（白场图）
             -> tone_map（亮度拉平 + 高光压缩）
             -> 文字保护（字芯与低对比晕边回落到基线线性映射）
             -> repair_holes（补不涉及文字的破洞）
             -> uint8 灰度图 + 统计信息
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from ..algorithms.masks import PageMasks, compute_masks
from ..algorithms.repair import repair_holes
from ..algorithms.tone import apply_text_guard, soft_text_weight, text_guard_weight, tone_map
from ..algorithms.white_point import (
    PERCENTILE_MODES,
    bg_field,
    local_bg_stat,
    local_envelope,
    resolve_white_field,
    whitest_region,
)
from ..config import Curve, DEFAULT_PARAMS, Params
from ..morphology import dilate, disk, gaussian_blur, square

__all__ = ["level_page", "text_guard_mask"]


def level_page(
    rgb: np.ndarray,
    params: Optional[Params] = None,
    pre: Optional[PageMasks] = None,
    curve: Optional[Curve] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """处理一页，返回 (输出 uint8 灰度图, 白边掩膜, 统计信息)。

    :param rgb:    页面主位图（RGB uint8）
    :param params: 算法参数，默认使用 config.DEFAULT_PARAMS
    :param pre:    已算好的掩膜，用于同一页多方案对比时避免重复计算
    :param curve:  覆盖色调映射曲线（默认取 params.curve()）
    """
    params = params or DEFAULT_PARAMS
    masks = pre if pre is not None else compute_masks(rgb, params)
    gray, margin, ink, holes, paper = masks

    # 1) 基线白点：纸面最白连通区域的均值
    wp, area_frac = whitest_region(gray, paper, params.min_area_frac, params.outlier_k)
    if params.wp_mode in PERCENTILE_MODES:
        q = int(params.wp_mode[1:])
        wp = float(np.percentile(gray[paper], q)) if paper.any() else wp

    # 2) 黑场与纸面亮度（判定是否需要兜底白场）
    non_margin_med = float(np.median(gray[~margin])) if (~margin).any() else 200.0
    paper_med = float(np.median(gray[paper])) if paper.any() else non_margin_med
    ink_gray = gray[ink]
    bp = float(np.percentile(ink_gray, 50)) if ink_gray.size > 50 else non_margin_med * 0.4
    paper_level = _paper_level(wp, paper_med, bp)
    wp_map, wp_full, mode = resolve_white_field(masks, wp, paper_med, paper_level, params)

    # 3) 映射曲线
    curve = curve or params.curve()
    if curve and curve[0] == "knee_auto":
        q, amp = int(curve[1]), float(curve[2])
        pq = float(np.percentile(gray[paper], q)) if paper.any() else paper_med
        dk = (wp - pq) / max(wp - bp, 30.0)
        curve = ("knee", min(max(dk, 0.05), 0.90), amp)

    # 4) 可选：局部背景抬平（只抬不压，字周围衰减为 0）
    gray_before_lift = gray
    if params.lift:
        if isinstance(params.lift, (tuple, list)):
            env = local_bg_stat(gray, paper, win=int(params.lift[1]), k=float(params.lift[2]))
        else:
            env = local_envelope(gray, paper, iters=int(params.lift))
        corr = np.maximum(wp - env, 0.0)
        if params.ink_protect > 0:
            prot = dilate(ink, square(int(params.ink_protect)))
            corr = corr * np.clip(
                1.0 - gaussian_blur(prot.astype(np.float32), 2.0), 0.0, 1.0
            )
        gray = np.clip(gray + corr, 0, 255)

    # 5) 亮度拉平
    out = tone_map(gray, margin, wp_map, bp, curve)

    # 6) 文字保护：字芯与抗锯齿边缘回落到「基线白场 + 线性映射」
    if params.text_guard:
        if isinstance(params.text_guard, (tuple, list)):
            weight = soft_text_weight(gray_before_lift, masks,
                                      threshold=float(params.text_guard[1]))
        else:
            guard = text_guard_mask(gray_before_lift, masks, params)
            weight = text_guard_weight(guard, int(params.text_guard), params.text_guard_sigma)
        # 基线必须用全局白场 Wp_full：用局部白场做基线会让字的映射随邻域漂移。
        baseline = tone_map(gray_before_lift, margin, wp_full, bp, ("linear",))
        out = apply_text_guard(out, baseline, weight)

    # 7) 破洞修补
    filled = skipped = 0
    if params.repair_holes_enabled:
        out, filled, skipped = repair_holes(out, gray, margin, ink, holes)

    info = dict(
        Wp=float(wp),
        area=float(area_frac),
        bp=float(bp),
        paper_med=float(paper_med),
        paper_level=paper_level,
        mode=mode,
        margin=float(margin.mean()),
        ink=float(ink.mean()),
        hole_px=int(holes.sum()),
        filled=filled,
        skipped=skipped,
        width=int(gray.shape[1]),
        height=int(gray.shape[0]),
    )
    return out, margin, info


def text_guard_mask(gray: np.ndarray, masks: PageMasks, params: Params) -> np.ndarray:
    """文字保护掩膜 = 墨迹掩膜 ∪ 墨迹近旁的低对比洇墨晕边。

    全局墨迹阈值（ink_ratio × 中位）只套得住浓墨字芯；木版印刷的洇墨晕边
    比纸面暗得不多（正文页实测仅暗 20-60 灰阶），会被 gamma 漂白成
    「文字偏浅」。这里以局部纸面亮度场（bg_field，多尺度）为参照，比纸面暗
    超过 text_dev_thr 的像素纳入保护掩膜，按基线线性映射保留其浓度。

    两点约束：
    * 距离上限 text_dev_radius：距墨迹超过该距离的低对比区域是背面透印的
      污渍而非笔画晕边，仍按纸面漂白——在「字迹清晰」与「污渍清洁」间取平衡。
    * 晕边像素本身被算作 strict_paper，会把亮度场拉低到与晕边持平，故迭代去
      污染（每轮把上一轮候选及其 2px 邻域从参照样本中剔除后重估）；迭代上限
      2 轮，兼顾收敛与「把纸面阴影也当墨」的过保护风险。参照样本耗尽时自动
      退回仅墨迹掩膜。
    """
    if params.text_dev_thr <= 0:
        return masks.ink
    pm = masks.strict_paper.astype(np.float32)
    cand = np.zeros_like(masks.ink)
    for _ in range(2):
        bg = bg_field(gray, pm, params.bg_wins)
        cand = (bg - gray) > params.text_dev_thr
        cand &= ~masks.margin
        pm_new = masks.strict_paper & ~dilate(cand, square(2))
        if np.array_equal(pm_new, pm):
            break
        pm = pm_new.astype(np.float32)
    if params.text_dev_radius > 0:
        # 只保留墨迹近旁的晕边；远离墨迹的透印污渍仍按纸面漂白
        cand &= dilate(masks.ink, disk(params.text_dev_radius))
    return masks.ink | cand


def _paper_level(wp: float, paper_med: float, bp: float) -> float:
    """纸面中位灰度映射后的亮度值；低于 safeguard_level 说明该页强色偏。"""
    return 255.0 * (1.0 - min((wp - paper_med) / max(wp - bp, 30.0), 1.0))
