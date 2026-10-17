# -*- coding: utf-8 -*-
"""PyInstaller 打包入口（必须放在包外）。

不要用 `classicsclean/__main__.py` 当入口：PyInstaller 会把它当作顶层脚本执行，
包内相对导入（`from ..config import ...`）会报
"attempted relative import with no known parent package"。
这里以「外部脚本 + pathex=src」的方式启动，包结构才是完整的。
"""
import sys

from classicsclean.apps.cli import main

if __name__ == "__main__":
    sys.exit(main())
