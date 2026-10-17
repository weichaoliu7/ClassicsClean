# -*- coding: utf-8 -*-
"""独立运行全书终检（不经过 pytest），并把报告写到 reports/。

用法：
    python scripts/run_acceptance.py                      # 用 samples/ 里的真书
    python scripts/run_acceptance.py --source a.pdf       # 指定源 PDF
    python scripts/run_acceptance.py --rebuild            # 忽略已有成品，重新出片

产物：
    reports/acceptance-pages.csv   逐页指标
    reports/acceptance.json        汇总 + 逐页数据（机读）
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from acceptance.acceptance import (  # noqa: E402
    ACCEPTANCE_CSV,
    ACCEPTANCE_JSON,
    OUTPUT_SUFFIX,
    AcceptanceTargets,
    check,
    ensure_output,
    find_source,
    run,
    write_reports,
)
from classicsclean.config import DEFAULT_PARAMS  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="ClassicsClean 全书终检")
    ap.add_argument("--source", help="源 PDF（默认自动在 samples/ 中查找）")
    ap.add_argument("--output", help="成品 PDF（默认自动在 samples/ 中查找/生成）")
    ap.add_argument("--db", type=float, default=DEFAULT_PARAMS.jp2_db, help="JPEG2000 编码质量(dB)")
    ap.add_argument("--rebuild", action="store_true", help="忽略已有成品样本，强制重新出片")
    args = ap.parse_args()

    source = Path(args.source) if args.source else find_source()
    if source is None:
        print("samples/ 下没有源 PDF，无法做全书终检")
        return 1

    if args.rebuild and not args.output:
        stale = source.with_name(f"{source.stem}{OUTPUT_SUFFIX}.pdf")
        if stale.is_file():
            stale.unlink()
            print(f"已删除旧成品样本：{stale.name}")

    output = Path(args.output) if args.output else ensure_output(source, args.db)
    rows, summary = run(source, output, args.db)
    write_reports(rows, summary, ACCEPTANCE_CSV, ACCEPTANCE_JSON)

    print("\n===== 全书终检汇总 =====")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"\n逐页数据 -> {ACCEPTANCE_CSV}")
    print(f"机读汇总 -> {ACCEPTANCE_JSON}")

    fails = check(summary, AcceptanceTargets())
    if fails:
        print("\n未通过阈值校验：")
        for f in fails:
            print("  -", f)
        return 1
    print("\n阈值校验：全部通过 ✅")
    return 0


if __name__ == "__main__":
    sys.exit(main())
