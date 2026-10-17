# -*- coding: utf-8 -*-
"""全书终检（pytest 包装）—— 真书 48 页的统计口径验收。

跑法：
    pytest -m slow
    python scripts/run_acceptance.py        # 独立运行（推荐，直达报告）

断言来自 `acceptance.AcceptanceTargets`，即本项目对外承诺的质量底线。
本用例会把逐页数据写到 reports/，供 scripts/run_tests.py 汇总进测试报告。
"""
from __future__ import annotations

from pathlib import Path

import pytest

from .acceptance import (
    ACCEPTANCE_CSV,
    ACCEPTANCE_JSON,
    AcceptanceTargets,
    check,
    ensure_output,
    find_source,
    run,
    write_reports,
)

pytestmark = pytest.mark.slow


def test_full_book_acceptance():
    """真书验收：背景够白、字没被动、成品与算法一致。"""
    source = find_source()
    if source is None:
        pytest.skip("samples/ 下没有源 PDF")
    output = ensure_output(source)

    rows, summary = run(source, output)
    write_reports(rows, summary, ACCEPTANCE_CSV, ACCEPTANCE_JSON)

    print("\n===== 全书终检汇总 =====")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    fails = check(summary, AcceptanceTargets())
    assert not fails, "全书终检未通过：\n  - " + "\n  - ".join(fails)
