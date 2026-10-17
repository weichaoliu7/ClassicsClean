# -*- coding: utf-8 -*-
"""平衡检测的 pytest 入口（标 slow，默认跳过，`pytest -m slow` 时执行）。"""
import pytest

from .balance import check, find_source, run


@pytest.mark.slow
def test_stain_text_balance():
    source = find_source()
    assert source is not None, "samples/ 下找不到源 PDF"
    _, summary = run(source)
    fails = check(summary)
    assert not fails, "平衡检测未通过：\n  " + "\n  ".join(fails)
