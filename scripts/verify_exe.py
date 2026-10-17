# -*- coding: utf-8 -*-
"""成品 exe 验收：环境自检 + 真实转码冒烟 + 输出格式检查。

与单元测试的区别：这里验证的是**打包后的产物**——依赖是否打全、JPEG2000
编解码在 exe 内是否可用、出片格式是否仍是 8bit 灰度 JPXDecode。

用法：
    python scripts/verify_exe.py [exe路径]
产物：
    reports/exe-selftest.json   exe 自报的环境自检结果
    reports/exe-verify.json     本脚本的验收结论
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

import pikepdf  # noqa: E402
from conftest import make_scan_page, write_scan_pdf  # noqa: E402

REPORTS = ROOT / "reports"
SELFTEST_JSON = REPORTS / "exe-selftest.json"
VERIFY_JSON = REPORTS / "exe-verify.json"


def main() -> int:
    exe = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "dist" / "ClassicsClean.exe"
    if not exe.is_file():
        print(f"找不到 exe：{exe}")
        return 1
    REPORTS.mkdir(parents=True, exist_ok=True)

    checks: dict = {"exe": exe.name, "体积MB": round(exe.stat().st_size / 1048576, 2)}
    ok = True

    # 1) 自检
    t0 = time.time()
    proc = subprocess.run([str(exe), "--selftest", "--report", str(SELFTEST_JSON)],
                          capture_output=True, text=True, encoding="utf-8", errors="replace")
    selftest = json.loads(SELFTEST_JSON.read_text(encoding="utf-8")) if SELFTEST_JSON.is_file() else {}
    checks["自检退出码"] = proc.returncode
    checks["自检结论"] = selftest.get("verdict", "unknown")
    checks["自检失败项"] = selftest.get("failed", [])
    checks["自检耗时秒"] = round(time.time() - t0, 1)
    ok &= proc.returncode == 0 and selftest.get("verdict") == "pass"

    # 2) 真实转码冒烟
    with tempfile.TemporaryDirectory(prefix="ccverify_") as tmp:
        tmp = Path(tmp)
        src = write_scan_pdf(tmp / "input.pdf", [make_scan_page(seed=3), make_scan_page(seed=5)])
        dst = tmp / "output.pdf"
        report = tmp / "report.json"
        t0 = time.time()
        proc = subprocess.run(
            [str(exe), src, "-o", str(dst), "--validate", "--report", str(report)],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        checks["转码退出码"] = proc.returncode
        checks["转码耗时秒"] = round(time.time() - t0, 1)
        ok &= proc.returncode == 0 and dst.is_file()

        if dst.is_file():
            with pikepdf.open(str(dst)) as pdf:
                checks["输出页数"] = len(pdf.pages)
                gray = all(str(p.Resources.XObject["/Im0"].ColorSpace) == "/DeviceGray" for p in pdf.pages)
                jpx = all(str(p.Resources.XObject["/Im0"].Filter) == "/JPXDecode" for p in pdf.pages)
                checks["输出灰度"] = gray
                checks["输出JPEG2000"] = jpx
                ok &= len(pdf.pages) == 2 and gray and jpx
            checks["输出体积KB"] = round(dst.stat().st_size / 1024, 1)
        if report.is_file():
            data = json.loads(report.read_text(encoding="utf-8"))
            rows = (data["results"][0] or {}).get("validation") or []
            checks["回读PSNR最低dB"] = round(min((r["psnr"] for r in rows), default=0.0), 2)
            ok &= bool(rows) and checks["回读PSNR最低dB"] > 35.0

    if proc.returncode != 0:
        print(proc.stdout[-2000:])
        print(proc.stderr[-2000:])

    checks["结论"] = "pass" if ok else "fail"
    VERIFY_JSON.write_text(json.dumps(checks, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n===== exe 验收 =====")
    for k, v in checks.items():
        print(f"  {k}: {v}")
    print(f"\n机读结论 -> {VERIFY_JSON}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
