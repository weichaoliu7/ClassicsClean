# -*- coding: utf-8 -*-
"""【第 4 层 · 入口层】同一套算法的两种入口，不重复任何算法逻辑。

    cli.py  命令行（批处理 / 无头 / 自检）
    gui.py  图形界面（拖拽 / 选择文件）

两个入口都只调用编排层的 `process_pdf`。GUI 依赖 tkinter，仅在真正启动时导入。
"""
from .cli import main

__all__ = ["main", "launch"]


def launch() -> int:
    """启动图形界面（延迟导入 tkinter，命令行模式下不会触碰 GUI 依赖）。"""
    from .gui import launch as _launch

    return _launch()
