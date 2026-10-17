# -*- coding: utf-8 -*-
"""【第 2 层 · 算法】色调映射与文字保护。

输入：灰度图 + 白场图 + 黑场 bp + 映射曲线
输出：uint8 灰度图
边界：纸面可以任意压白；**笔画必须逐像素等于基线线性映射**（红线 2）

核心不变式（定稿版红线）：
  **笔画及其抗锯齿边缘的输出必须逐像素等于「基线全局白点 + 线性映射」的结果。**
  纸面可以任意拉白，字一个灰阶都不许动。文字保护区的内圈权重硬取 1，
  且必须使用基线白场 Wp_full（局部白场会让字的映射随邻域漂移）。
"""
from __future__ import annotations

from typing import Tuple, Union

import numpy as np

from ..config import Curve, Params
from ..morphology import dilate, gaussian_blur, square
from .masks import PageMasks
from .white_point import bg_field

__all__ = ["tone_map", "soft_text_weight", "text_guard_weight", "apply_text_guard"]


def tone_map(
    gray: np.ndarray,
    margin: np.ndarray,
    wp_map: np.ndarray,
    bp: float,
    curve: Curve,
) -> np.ndarray:
    """把「相对白场的偏离量」映射为输出灰度。

    d = (Wp - gray) / (Wp - bp)，截断到 [0, 1]；文本保持连续灰阶，不做二值化。
      ("linear",)         t = d
      ("gamma", g)        t = d ** g            高光端压缩，残留底灰被推向白
      ("knee", dk, amp)   折线：背景段压进 [255, 255-amp]，文本段线性
    """
    rng = np.maximum(wp_map - bp, 30.0)
    d = np.clip(np.maximum(wp_map - gray, 0.0) / rng, 0.0, 1.0)
    kind = curve[0]
    if kind == "linear":
        t = d
    elif kind == "gamma":
        t = d ** float(curve[1])
    elif kind == "knee":
        dk = min(max(float(curve[1]), 1e-6), 0.98)
        amp = float(curve[2]) / 255.0
        t = np.where(d <= dk, d * amp / dk, amp + (d - dk) * (1.0 - amp) / max(1.0 - dk, 1e-6))
    else:
        raise ValueError(f"unknown curve: {curve}")
    out = 255.0 * (1.0 - np.clip(t, 0.0, 1.0))
    out[margin] = 255.0
    return np.clip(out, 0, 255).astype(np.uint8)


def soft_text_weight(
    gray: np.ndarray,
    masks: PageMasks,
    threshold: float = 15.0,
    erode_px: int = 2,
    wins=(31, 61, 121),
) -> np.ndarray:
    """按「偏离局部纸面的幅度」软判定某像素是不是字。

    dev 大 = 明显暗于周围纸面 = 笔画（含抗锯齿边缘）-> 权重 1；
    dev 小 = 与纸面同色（黄斑/霉渍/纸纹）-> 权重 0，按背景压白。
    实测该策略会残留灰边（std 升到 28.4），所以默认走 text_guard_weight 的硬保护，
    此处保留供算法研究与回归对比使用。
    """
    strict = masks.strict_paper
    env = bg_field(gray, strict, wins)
    dev = np.maximum(env - gray, 0.0)
    w = np.clip(dev / max(float(threshold), 1e-6), 0.0, 1.0)
    return w * w * (3.0 - 2.0 * w)  # smoothstep，避免硬边


def text_guard_weight(
    ink: np.ndarray,
    radius: int,
    sigma: float = 1.5,
    feather: int = 2,
) -> np.ndarray:
    """文字保护区权重：内圈严格 1（字芯与抗锯齿边缘逐像素不变），外圈羽化到 0。

    用 np.maximum(inner, blur(outer)) 保证内圈权重恒为 1 —— 若对内圈也做高斯
    羽化，权重永远到不了 1，字会被轻微改动。
    """
    inner = dilate(ink, square(int(radius))).astype(np.float32)
    outer = dilate(ink, square(int(radius) + int(feather))).astype(np.float32)
    return np.clip(
        np.maximum(inner, gaussian_blur(outer, float(sigma))),
        0.0,
        1.0,
    )


def apply_text_guard(
    out: np.ndarray,
    out_base: np.ndarray,
    weight: np.ndarray,
) -> np.ndarray:
    """按权重混合「基线映射结果」与「本页映射结果」。"""
    mixed = out_base.astype(np.float32) * weight + out.astype(np.float32) * (1.0 - weight)
    return np.clip(mixed, 0, 255).astype(np.uint8)
