# -*- coding: utf-8 -*-
"""全局配置：算法固化参数与编码参数。

本文件是整个项目唯一的「参数来源」。所有默认值均来自定稿算法的实测结论，
修改任一数值都会改变输出 PDF 的观感，改动前请阅读 docs/architecture.md。
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Tuple, Union

__all__ = ["Params", "DEFAULT_PARAMS", "VERSION", "APP_NAME", "Curve",
           "PERCENTILE_MODES", "WHITE_POINT_MODES"]

APP_NAME = "ClassicsClean"
VERSION = "1.1.0"

# 纸面分位白场模式（仅供算法研究与回归对比，定稿不使用）
PERCENTILE_MODES = ("p50", "p60", "p65", "p75")
# 白场模式的合法取值：`local` 局部纸面亮度场、`whitest` 全局最白区、以及分位模式
WHITE_POINT_MODES = ("local", "whitest") + PERCENTILE_MODES

# 色调映射曲线的三种形态：
#   ("linear",)           线性映射（v11 基线方案）
#   ("gamma", g)          高光压缩，g 越大背景越白
#   ("knee", dk, amp)     折线映射：背景段压进 [255, 255-amp]
Curve = Union[Tuple[str], Tuple[str, float], Tuple[str, float, float]]


@dataclass(frozen=True)
class Params:
    """一次处理任务所需的全部参数（不可变，用 dataclasses.replace 派生变体）。"""

    # ---------------- 掩膜分割 ----------------
    margin_thr: float = 235.0
    """纯白边种子亮度阈值：与图像边缘连通的亮区视为扫描白边，全程不参与统计。"""

    ink_ratio: float = 0.62
    """墨迹判定阈值：gray < ink_ratio * median(非白边灰度)。"""

    hole_thr: float = 225.0
    """破洞亮度阈值：图像内部（不与白边连通）的孤立亮块。"""

    hole_min_area: int = 30
    """破洞最小面积（像素）。"""

    hole_max_frac: float = 0.03
    """破洞最大面积（占整页比例）。"""

    # ---------------- 白场估计 ----------------
    min_area_frac: float = 0.008
    """最白区域面积下限 = 整页的 0.8%（自适应放大候选集直到最大连通分量达标）。"""

    wp_mode: str = "local"
    """白场模式：local=逐点局部纸面亮度场；whitest=全局最白区域均值；p50/p65=纸面分位。"""

    bg_wins: Tuple[int, ...] = (31, 61, 121)
    """局部白场的多尺度窗口（由粗到细逐级细化）。"""

    outlier_k: float = 2.5
    """白区均值上限 = 纸面均值 + outlier_k × 纸面标准差；超过则视为破洞/高光噪点。"""

    safeguard_level: float = 200.0
    """纸面亮度低于该值时触发兜底白场（深红封面等强色偏页）。"""

    safeguard_win: int = 61
    """兜底白场统计窗口边长。"""

    safeguard_k: float = 0.8
    """兜底白场 = 局部均值 + safeguard_k × 局部标准差。"""

    # ---------------- 色调映射 ----------------
    gamma: float = 3.0
    """高光压缩指数（1.0 = 线性，不压缩）。"""

    text_guard: Union[int, Tuple[str, float]] = 1
    """文字保护：int=笔画膨胀半径（像素迭代次数）；("dev", T)=按偏离局部纸面的幅度软判定。"""

    text_guard_sigma: float = 1.5
    """文字保护区外圈的高斯羽化 sigma（内圈权重恒为 1，保证字芯逐像素不变）。"""

    text_dev_thr: float = 15.0
    """文字保护的低对比扩展阈值（灰阶，0 = 关闭）。

    木版印刷的洇墨晕边只比纸面暗 20-60 灰阶，全局墨迹阈值
    （ink_ratio × 中位）套不住，会被 gamma 漂白成「文字偏浅」。
    这里以局部纸面亮度场（bg_field）为参照：比局部纸面暗超过该阈值的
    像素一律并入文字硬保护区，按基线线性映射保留其浓度。阈值需明显
    高于扫描噪声幅度（约 3-6 灰阶），否则纸面纹理会被误保护。"""

    text_dev_radius: int = 12
    """低对比晕边保护的距离上限（像素，0 = 不限距离）。

    距墨迹超过该距离的低对比区域是背面透印的污渍（而非笔画晕边），
    仍按纸面漂白。该值在「字迹清晰」与「透印污渍清洁」之间取平衡：
    调大则保留更多晕边（字更浓但污渍也更明显），调小则相反。"""

    # ---------------- 可选增强（默认关闭） ----------------
    lift: int = 0
    """局部背景抬平迭代次数，0 = 关闭（已由局部白场取代）。"""

    ink_protect: int = 3
    """lift 生效时的文字保护半径（抬平量在该半径内衰减为 0）。"""

    repair_holes_enabled: bool = True
    """是否修补不涉及文字的破洞。"""

    # ---------------- 编码 ----------------
    jp2_db: float = 52.0
    """JPEG2000 编码质量（dB）。52dB 实测：字核心编码噪声 <= 16 灰阶、PSNR >= 50dB。"""

    jp2_num_resolutions: int = 6
    """JPEG2000 分解层数（小图会自动下调，避免编码失败）。"""

    # ---------------- 运行 ----------------
    validate: bool = False
    """出片后是否回读校验（耗时约等于处理时间，默认关闭）。"""

    def __post_init__(self) -> None:
        """参数校验：宁可在构造时报错，也不要带着非法参数跑完一本书。"""
        if self.gamma <= 0:
            raise ValueError(f"gamma 必须为正数，当前 {self.gamma}")
        if self.wp_mode not in WHITE_POINT_MODES:
            raise ValueError(f"wp_mode 必须是 {WHITE_POINT_MODES} 之一，当前 {self.wp_mode!r}")
        if not 0 < self.min_area_frac <= 1:
            raise ValueError(f"min_area_frac 必须落在 (0, 1]，当前 {self.min_area_frac}")
        if not 20.0 <= self.jp2_db <= 80.0:
            raise ValueError(f"jp2_db 应落在 [20, 80] dB，当前 {self.jp2_db}")
        if not 0.0 < self.ink_ratio < 1.0:
            raise ValueError(f"ink_ratio 必须落在 (0, 1)，当前 {self.ink_ratio}")
        if self.text_dev_thr < 0 or self.text_dev_thr > 100:
            raise ValueError(f"text_dev_thr 必须落在 [0, 100]，当前 {self.text_dev_thr}")
        if self.text_dev_radius < 0:
            raise ValueError(f"text_dev_radius 不能为负，当前 {self.text_dev_radius}")

    def curve(self) -> Curve:
        """当前的色调映射曲线。"""
        if abs(self.gamma - 1.0) < 1e-9:
            return ("linear",)
        return ("gamma", float(self.gamma))

    def with_db(self, db: float) -> "Params":
        """返回仅修改编码质量的副本（GUI 质量档位使用）。"""
        return replace(self, jp2_db=float(db))


DEFAULT_PARAMS = Params()
