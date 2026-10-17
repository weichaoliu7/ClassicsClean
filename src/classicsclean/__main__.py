# -*- coding: utf-8 -*-
"""`python -m classicsclean` 入口：完全等价于命令行主程序 `main()`。"""
import sys

from .apps.cli import main

if __name__ == "__main__":
    sys.exit(main())
