# -*- coding: utf-8 -*-
"""一键自动化测试 + 报告生成。

用法：
    python scripts/run_tests.py                 # 只跑秒级单元测试
    python scripts/run_tests.py --scope full    # 单元测试 + 全书终检（分钟级）
    python scripts/run_tests.py --exe           # 额外对 dist/ClassicsClean.exe 做冒烟验证
    python scripts/run_tests.py --scope full --rebuild-sample   # 重建成品样本后再终检

产物（reports/，本脚本自动生成，请勿手改）：
    test-report.md          人读的测试报告（本文件由脚本生成）
    test-report.json        机读结果，供 CI / 回归比对
    junit.xml               合并后的 JUnit 结果，可直接喂给 CI
    junit-unit.xml          单元测试原始结果
    junit-acceptance.xml    全书终检原始结果（仅 --scope full）
    acceptance-pages.csv    全书终检逐页数据（仅 --scope full）
    acceptance.json         全书终检汇总（仅 --scope full）
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "src", ROOT / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

REPORTS = ROOT / "reports"
JUNIT = REPORTS / "junit.xml"
JUNIT_UNIT = REPORTS / "junit-unit.xml"
JUNIT_ACCEPT = REPORTS / "junit-acceptance.xml"
REPORT_MD = REPORTS / "test-report.md"
REPORT_JSON = REPORTS / "test-report.json"
ACCEPTANCE_CSV = REPORTS / "acceptance-pages.csv"
ACCEPTANCE_JSON = REPORTS / "acceptance.json"

# 红线 → 守护它的单元测试用例
RED_LINES: List[Tuple[str, List[str]]] = [
    ("红线 1 · 纯白扫描边输出恒为 255",
     ["test_tone_map_margin_is_pure_white", "test_output_basic_properties",
      "test_margin_stays_white_after_repair", "test_safeguard_engaged_on_dark_page"]),
    ("红线 2 · 笔画及抗锯齿边缘逐像素不变",
     ["test_text_guard_keeps_strokes_identical", "test_strokes_are_pixel_identical_to_baseline",
      "test_text_guard_weight_inner_is_exactly_one"]),
    ("红线 3 · 白场只统计纸面（不与白边/墨迹/破洞重叠）",
     ["test_whitest_region_never_touches_excluded_areas", "test_whitest_region_value_is_paper_tone"]),
    ("红线 4 · 碰字的破洞一律不补",
     ["test_hole_touching_text_is_skipped", "test_hole_away_from_text_is_filled"]),
]


# --------------------------------------------------------------------------- 环境
def environment() -> Dict:
    import importlib

    info = {
        "时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "操作系统": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "Python": platform.python_version(),
        "解释器": sys.executable,
        "依赖": {},
    }
    for name in ("numpy", "cv2", "pymupdf", "pikepdf", "PIL", "pytest"):
        try:
            mod = importlib.import_module(name)
            info["依赖"][name] = str(getattr(mod, "__version__", "ok"))
        except Exception as exc:                        # pragma: no cover
            info["依赖"][name] = f"缺失（{exc}）"
    return info


# --------------------------------------------------------------------------- 跑 pytest
def run_pytest(junit_path: Path, target: str, include_slow: bool) -> int:
    REPORTS.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "pytest", "--junitxml", str(junit_path), "-q"]
    if include_slow:
        # 覆盖 pytest.ini 里默认的 `-m "not slow"`，让全书终检也跑起来
        cmd += ["-o", "addopts=-ra --strict-markers"]
    cmd.append(target)
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    output = (proc.stdout or "") + (proc.stderr or "")
    print(output[-6000:] if len(output) > 6000 else output)
    return proc.returncode


def parse_junit(path: Path) -> Dict:
    """解析 JUnit XML，得到总览与按模块分组的明细。"""
    if not path.is_file():
        return {"summary": {}, "groups": {}, "cases": {}}
    root = ET.parse(path).getroot()
    suite = root if root.tag == "testsuite" else root.find("testsuite")
    cases = suite.findall("testcase") if suite is not None else []

    groups: Dict[str, Dict] = {}
    results: Dict[str, Dict] = {}
    total = {"passed": 0, "failed": 0, "skipped": 0, "error": 0}
    for c in cases:
        module = (c.get("classname") or "").replace(".", "/")
        name = c.get("name") or ""
        dur = float(c.get("time") or 0.0)
        status, detail = "passed", ""
        for tag, label in (("failure", "failed"), ("error", "error"), ("skipped", "skipped")):
            node = c.find(tag)
            if node is not None:
                status = label
                detail = ((node.get("message") or "").strip().splitlines() or [""])[0]
                break
        total[status] += 1
        g = groups.setdefault(module, {"cases": 0, "passed": 0, "failed": 0,
                                       "skipped": 0, "error": 0, "time": 0.0})
        g["cases"] += 1
        g[status] += 1
        g["time"] += dur
        results[f"{module}::{name}"] = {"status": status, "time": dur, "detail": detail}

    return {
        "summary": {
            "total": len(cases),
            "passed": total["passed"],
            "failed": total["failed"] + total["error"],
            "skipped": total["skipped"],
            "time": round(float(suite.get("time") or 0.0), 2) if suite is not None else 0.0,
        },
        "groups": groups,
        "cases": results,
    }


def merge_junit(parts: List[Dict]) -> Dict:
    """合并多份 JUnit 解析结果。"""
    merged: Dict = {"summary": {"total": 0, "passed": 0, "failed": 0, "skipped": 0, "time": 0.0},
                    "groups": {}, "cases": {}}
    for part in parts:
        for key in ("total", "passed", "failed", "skipped", "time"):
            merged["summary"][key] += part.get("summary", {}).get(key, 0)
        for module, g in part.get("groups", {}).items():
            target = merged["groups"].setdefault(
                module, {"cases": 0, "passed": 0, "failed": 0, "skipped": 0, "error": 0, "time": 0.0})
            for k, v in g.items():
                target[k] += v
        merged["cases"].update(part.get("cases", {}))
    merged["summary"]["time"] = round(merged["summary"]["time"], 2)
    return merged


def write_merged_junit(parts: List[Path], dest: Path) -> None:
    """把多份 JUnit XML 合并为一个文件（供 CI 消费）。"""
    suites = [ET.parse(p).getroot() for p in parts if p.is_file()]
    if not suites:
        return
    base = suites[0]
    if base.tag != "testsuite":
        found = base.find("testsuite")
        base = found if found is not None else base
    for other in suites[1:]:
        root = other if other.tag == "testsuite" else other.find("testsuite")
        if root is None:
            continue
        for case in root.findall("testcase"):
            base.append(case)
    dest.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(base).write(dest, encoding="utf-8", xml_declaration=True)


# --------------------------------------------------------------------------- exe 冒烟
def smoke_test_exe() -> Optional[Dict]:
    """对打包好的 exe 做冒烟验证（自检 + 真实转码 + 输出格式检查）。"""
    exe = ROOT / "dist" / "ClassicsClean.exe"
    if not exe.is_file():
        return None
    proc = subprocess.run([sys.executable, str(ROOT / "scripts" / "verify_exe.py")],
                          cwd=str(ROOT), capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    print(proc.stdout[-3000:] if proc.stdout else "")
    data_file = REPORTS / "exe-verify.json"
    if data_file.is_file():
        return json.loads(data_file.read_text(encoding="utf-8"))
    return {"结论": "fail", "备注": "exe 验收未产出报告", "退出码": proc.returncode}


# --------------------------------------------------------------------------- 报告
def _md_table(headers: List[str], rows: List[List]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out)


def build_markdown(env: Dict, scope: str, junit: Dict, acceptance: Optional[Dict],
                   exe: Optional[Dict], ok: bool, wall: float) -> str:
    s = junit["summary"]
    lines: List[str] = []
    lines.append("# ClassicsClean 测试报告")
    lines.append("")
    lines.append(f"**结论：{'通过 ✅' if ok else '未通过 ❌'}**　"
                 f"生成时间 {env['时间']}　测试范围 `{scope}`　总耗时 {wall:.1f}s")
    lines.append("")
    lines.append("> 本文件由 `python scripts/run_tests.py` 自动生成，请勿手工编辑。")
    lines.append("")

    lines.append("## 1. 运行环境")
    lines.append("")
    lines.append(_md_table(["项目", "值"],
                           [["操作系统", env["操作系统"]],
                            ["Python", env["Python"]],
                            ["依赖", "、".join(f"{k} {v}" for k, v in env["依赖"].items())]]))
    lines.append("")

    lines.append("## 2. 测试总览")
    lines.append("")
    lines.append(_md_table(["指标", "数值"],
                           [["用例总数", s["total"]],
                            ["通过", s["passed"]],
                            ["失败", s["failed"]],
                            ["跳过", s["skipped"]],
                            ["pytest 统计耗时 (s)", s["time"]]]))
    lines.append("")

    lines.append("## 3. 分层结果")
    lines.append("")
    rows = [[f"`{m}`", g["cases"], g["passed"], g["failed"], g["skipped"], f"{g['time']:.2f}"]
            for m, g in sorted(junit["groups"].items())]
    lines.append(_md_table(["测试模块", "用例", "通过", "失败", "跳过", "耗时(s)"], rows))
    lines.append("")

    failures = [(k, v) for k, v in junit["cases"].items() if v["status"] in ("failed", "error")]
    lines.append("## 4. 失败明细")
    lines.append("")
    if failures:
        lines.append(_md_table(["用例", "结论", "摘要"],
                               [[f"`{k}`", v["status"], (v["detail"] or "")[:120]]
                                for k, v in failures]))
    else:
        lines.append("无失败用例。")
    lines.append("")

    lines.append("## 5. 红线守护清单")
    lines.append("")
    rows = []
    for title, cases in RED_LINES:
        hit = [(c, junit["cases"][c]["status"]) for c in cases if c in junit["cases"]]
        missing = len(cases) - len(hit)
        verdict = "通过" if hit and all(st == "passed" for _, st in hit) else "未通过"
        if missing:
            verdict += f"（缺 {missing} 项）"
        rows.append([title, verdict, f"{len(hit)}/{len(cases)}"])
    if acceptance is not None:
        diff = acceptance["summary"].get("字核心算法差最大")
        rows.append(["红线 2 · 真书字核心算法差 == 0",
                     "通过" if diff == 0 else "未通过", f"实测 {diff}"])
    lines.append(_md_table(["红线", "结论", "覆盖"], rows))
    lines.append("")

    lines.append("## 6. 全书终检（真书）")
    lines.append("")
    if acceptance is None:
        lines.append("本次未运行（`--scope full` 才会跑，耗时分钟级）。")
    else:
        lines.append(_md_table(["指标", "值"], [[k, v] for k, v in acceptance["summary"].items()]))
        lines.append("")
        lines.append("**阈值校验**：" + ("全部通过" if not acceptance["fails"]
                                     else "存在失败项：" + "；".join(acceptance["fails"])))
        lines.append("")
        lines.append(f"逐页数据见 `reports/{ACCEPTANCE_CSV.name}`，"
                     f"机读汇总见 `reports/{ACCEPTANCE_JSON.name}`。")
    lines.append("")

    lines.append("## 7. 成品 exe 冒烟验证")
    lines.append("")
    if exe is None:
        lines.append("本次未运行（加 `--exe` 参数才会跑，或 dist/ClassicsClean.exe 不存在）。")
    else:
        lines.append(_md_table(["指标", "值"], [[k, v] for k, v in exe.items()]))
    lines.append("")

    lines.append("## 8. 产物清单")
    lines.append("")
    produced = [JUNIT, JUNIT_UNIT, REPORT_JSON, REPORT_MD]
    if scope == "full":
        produced += [JUNIT_ACCEPT, ACCEPTANCE_CSV, ACCEPTANCE_JSON]
    lines.append("\n".join(f"- `reports/{p.name}`" for p in produced))
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- 主流程
def main() -> int:
    ap = argparse.ArgumentParser(description="ClassicsClean 一键测试与报告生成")
    ap.add_argument("--scope", choices=["unit", "full"], default="unit",
                    help="unit=秒级单元测试；full=单元测试 + 全书终检")
    ap.add_argument("--exe", action="store_true", help="额外对 dist/ClassicsClean.exe 做冒烟验证")
    ap.add_argument("--rebuild-sample", action="store_true",
                    help="全书终检前先重建成品样本（忽略已有的 _清洁版.pdf）")
    args = ap.parse_args()

    REPORTS.mkdir(parents=True, exist_ok=True)
    env = environment()
    t0 = time.time()

    if args.rebuild_sample:
        sys.path.insert(0, str(ROOT / "tests"))
        from acceptance.acceptance import OUTPUT_SUFFIX, find_source

        source = find_source()
        if source is not None:
            stale = source.with_name(f"{source.stem}{OUTPUT_SUFFIX}.pdf")
            if stale.is_file():
                stale.unlink()
                print(f"已删除旧成品样本：{stale.name}")

    rc_unit = run_pytest(JUNIT_UNIT, "tests/unit", include_slow=False)
    rc_full = 0
    if args.scope == "full":
        rc_full = run_pytest(JUNIT_ACCEPT, "tests/acceptance", include_slow=True)

    parts = [parse_junit(JUNIT_UNIT)]
    if args.scope == "full":
        parts.append(parse_junit(JUNIT_ACCEPT))
    junit = merge_junit(parts)
    write_merged_junit([JUNIT_UNIT, JUNIT_ACCEPT] if args.scope == "full" else [JUNIT_UNIT], JUNIT)

    acceptance: Optional[Dict] = None
    if args.scope == "full" and ACCEPTANCE_JSON.is_file():
        sys.path.insert(0, str(ROOT / "tests"))
        from acceptance.acceptance import AcceptanceTargets, check

        data = json.loads(ACCEPTANCE_JSON.read_text(encoding="utf-8"))
        acceptance = {"summary": data["summary"], "fails": check(data["summary"], AcceptanceTargets())}
        print("\n===== 全书终检阈值校验 =====")
        print("  通过" if not acceptance["fails"] else "  " + "；".join(acceptance["fails"]))

    exe = smoke_test_exe() if args.exe else None
    wall = time.time() - t0

    ok = (rc_unit == 0 and rc_full == 0
          and (acceptance is None or not acceptance["fails"])
          and (exe is None or exe.get("结论") == "pass"))

    REPORT_MD.write_text(build_markdown(env, args.scope, junit, acceptance, exe, ok, wall),
                         encoding="utf-8")
    REPORT_JSON.write_text(
        json.dumps({"environment": env, "scope": args.scope,
                    "pytest_exit_code": {"unit": rc_unit, "acceptance": rc_full},
                    "junit": junit, "acceptance": acceptance, "exe": exe,
                    "wall_seconds": round(wall, 1), "verdict": "pass" if ok else "fail"},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n测试报告已生成：{REPORT_MD}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
