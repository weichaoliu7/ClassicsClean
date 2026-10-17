# -*- coding: utf-8 -*-
"""【第 3 层 · 编排层】把算法层串成流水线，是全项目唯一的 IO 边界使用者。

    page.py      单页流水线：掩膜 → 白场 → 色调映射 → 文字保护 → 破洞修补
    document.py  整本流水线：读源 PDF → 逐页 → 写 PDF → 可选回读校验
"""
from .document import PageReport, Result, ValidationRow, process_pdf, validate_output
from .page import level_page

__all__ = [
    "level_page",
    "process_pdf",
    "validate_output",
    "PageReport",
    "ValidationRow",
    "Result",
]
