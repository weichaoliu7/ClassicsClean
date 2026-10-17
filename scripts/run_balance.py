# -*- coding: utf-8 -*-
"""独立运行「污渍 / 字迹」平衡检测：跑真书、写报告、按阈值判合格。

用法：
    python scripts/run_balance.py                # 用默认参数，写 reports/balance.json
    python scripts/run_balance.py --rebuild      # 先重新生成成品再检测（确保与代码同步）

退出码 0 = 合格，1 = 不合格。
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from acceptance.balance import (  # noqa: E402
    BALANCE_JSON,
    check,
    find_source,
    run,
    write_report,
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="污渍/字迹平衡检测")
    parser.add_argument("--report", default=str(BALANCE_JSON), help="报告 JSON 路径")
    args = parser.parse_args(argv)

    source = find_source()
    if source is None:
        print("[错误] samples/ 下找不到源 PDF")
        return 1

    rows, summary = run(source)
    write_report(rows, summary, Path(args.report))
    print("\n" + "=" * 60)
    for k, v in summary.items():
        print(f"{k}: {v}")

    fails = check(summary)
    if fails:
        print("\n未通过平衡检测：")
        for f in fails:
            print(f"  - {f}")
        return 1
    print("\n平衡检测：全部通过 ✅")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
