# -*- coding: utf-8 -*-
"""【第 2 层 · 算法层】纯 numpy 进、纯 numpy 出，不接触文件、不接触界面。

    masks.py        掩膜分割：白边 / 墨迹 / 破洞 / 纸面
    white_point.py  白场估计：最白区 · 局部白场 · 兜底
    tone.py         色调映射 + 文字保护（红线 1、2 的守护者）
    repair.py       破洞修补（红线 4 的守护者）
"""
from .masks import PageMasks, compute_masks, hole_mask, white_margin_mask
from .repair import repair_holes
from .tone import apply_text_guard, soft_text_weight, text_guard_weight, tone_map
from .white_point import (
    PERCENTILE_MODES,
    bg_field,
    growth_ladder,
    local_bg_stat,
    local_envelope,
    resolve_white_field,
    safeguard_field,
    whitest_region,
)

__all__ = [
    "PageMasks",
    "compute_masks",
    "hole_mask",
    "white_margin_mask",
    "repair_holes",
    "apply_text_guard",
    "soft_text_weight",
    "text_guard_weight",
    "tone_map",
    "PERCENTILE_MODES",
    "bg_field",
    "growth_ladder",
    "local_bg_stat",
    "local_envelope",
    "resolve_white_field",
    "safeguard_field",
    "whitest_region",
]
