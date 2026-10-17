# -*- coding: utf-8 -*-
"""ClassicsClean —— 古籍扫描件白底清洁工具。

一句话目标：**纸面尽量拉白，笔画一个灰阶都不动。**

=========================== 代码架构（四层） ============================

    第 4 层  入口层  apps/
                        cli.py   命令行（批处理 / 无头 / 环境自检）
                        gui.py   图形界面（拖拽 / 选择文件）
                            │  两者都只调用第 3 层，不重复任何算法逻辑
    第 3 层  编排层  pipeline/
                        page.py      单页流水线
                        document.py  整本流水线（唯一的 IO 边界）
                            │
    第 2 层  算法层  algorithms/
                        masks.py        掩膜分割：白边 / 墨迹 / 破洞 / 纸面
                        white_point.py  白场估计：最白区 · 局部白场 · 兜底
                        tone.py         色调映射 + 文字保护
                        repair.py       破洞修补
                            │
    第 1 层  基础层  config.py    参数唯一来源（所有常量集中在此）
                    imaging.py   PDF ↔ 位图、灰度化、JPEG2000 编解码

============================= 单页数据流 ==============================

    源页面位图 (RGB uint8)
        │
        ├─→ compute_masks ────→ 白边 / 墨迹 / 破洞 / 纸面（四者互斥）
        ├─→ whitest_region ──→ 基线白点 Wp
        ├─→ resolve_white_field → 白场图 Wp_map（逐像素）
        ├─→ tone_map ────────→ 亮度拉平 + gamma 高光压缩（纸面推向白）
        ├─→ text_guard ──────→ 笔画回落到「基线白场 + 线性映射」（字不动）
        └─→ repair_holes ────→ 补不涉及文字的破洞
        │
    输出灰度图 (uint8) ──→ encode_jp2 ──→ 成品 PDF 页面

============================== 四条红线 ===============================

    1. 白边恒为 255，任何曲线都不许动               —— tone.tone_map
    2. 笔画逐像素等于「基线白场 + 线性映射」          —— tone.text_guard_weight
    3. 白场只统计纸面，与白边 / 墨迹 / 破洞重叠为 0   —— white_point.whitest_region
    4. 碰字的破洞一律不补                            —— repair.repair_holes

对外 API（GUI / CLI / 测试都只用这几个）：

    >>> from classicsclean import process_pdf
    >>> result = process_pdf("in.pdf", "out.pdf")
    >>> result.page_count, result.output_bytes
"""
from .config import APP_NAME, DEFAULT_PARAMS, VERSION, Params
from .pipeline import (
    PageReport,
    Result,
    ValidationRow,
    level_page,
    process_pdf,
    validate_output,
)

__version__ = VERSION

__all__ = [
    "APP_NAME",
    "VERSION",
    "Params",
    "DEFAULT_PARAMS",
    "level_page",
    "process_pdf",
    "validate_output",
    "PageReport",
    "ValidationRow",
    "Result",
]
