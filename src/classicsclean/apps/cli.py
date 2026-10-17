# -*- coding: utf-8 -*-
"""【第 4 层 · 入口】命令行：批处理 / 无头模式。

不带文件路径时启动图形界面，带文件路径时直接处理退出（适合批量与自动化）。
只调用 process_pdf，不重复任何算法逻辑。

用法：
    ClassicsClean.exe                          # 无参数 -> 启动图形界面
    ClassicsClean.exe in.pdf                   # 处理单个文件
    ClassicsClean.exe a.pdf b.pdf -o out.pdf   # 批量（批量时 -o 视为目录）
    ClassicsClean.exe in.pdf --db 58 --validate --report result.json
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

from ..config import DEFAULT_PARAMS, VERSION
from ..pipeline.document import Result, process_pdf

LOG = logging.getLogger("classicsclean")


def _is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def _default_output(src: str) -> str:
    p = Path(src)
    return str(p.with_name(f"{p.stem}_清洁版.pdf"))


def _setup_logging(log_file: Optional[str], quiet: bool) -> None:
    """日志配置。

    打包为窗口化 exe 时 stdout 可能为 None（GUI 子系统不继承控制台），
    此时自动落到日志文件，保证命令行模式永远可追溯。
    """
    handlers: List[logging.Handler] = []
    if not quiet and sys.stdout is not None:
        handlers.append(logging.StreamHandler(sys.stdout))
    if log_file or not handlers:
        path = Path(log_file) if log_file else Path(_default_output("ClassicsClean")).with_suffix(".log")
        path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(path, encoding="utf-8"))
        LOG.info("日志文件：%s", path)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
        force=True,
    )


def _progress(done: int, total: int, text: str) -> None:
    LOG.info("[%d/%d] %s", done, total, text)


REQUIRED_MODULES = ("numpy", "pymupdf", "pikepdf", "PIL")
"""命令行 / 算法必需依赖：任一缺失即无法工作。

形态学与滤波算子由 classicsclean.morphology 以纯 numpy 实现（历史上曾依赖
opencv-python，其二进制占 exe 体积一半以上，已整体移除），此处不再需要 cv2。
"""

OPTIONAL_MODULES = ("tkinter", "tkinterdnd2")
"""仅图形界面需要的依赖：缺失时自动降级为命令行模式，不算失败。"""


def _selftest(report: Optional[str]) -> int:
    """环境自检：在一台新电脑上先跑它，确认关键能力都可用。

    必需项（缺失即判失败）：第三方库导入、JPEG2000 往返、临时目录可写
    （单文件 exe 每次启动都要往 %TEMP% 解包）。
    可选项（缺失仅降级）：tkinter / tkinterdnd2 —— 只影响图形界面与拖拽。
    """
    import importlib
    import tempfile

    required: dict = {}
    optional: dict = {}

    for name in REQUIRED_MODULES:
        try:
            mod = importlib.import_module(name)
            required[name] = f"ok {getattr(mod, '__version__', '')}".strip()
        except Exception as exc:                       # pragma: no cover
            required[name] = f"FAIL {exc}"

    for name in OPTIONAL_MODULES:
        try:
            importlib.import_module(name)
            optional[name] = "ok（图形界面可用）" if name == "tkinter" else "ok（拖拽可用）"
        except Exception as exc:                       # pragma: no cover
            optional[name] = f"缺失，已降级（{exc}）"

    try:
        import numpy as np

        from ..imaging import decode_jp2, encode_jp2

        probe = np.full((64, 64), 200, np.uint8)
        decoded = decode_jp2(encode_jp2(probe, 52.0), probe.shape)
        err = int(np.abs(decoded.astype(int) - probe.astype(int)).max())
        required["jpeg2000"] = f"ok 往返最大误差={err}"
    except Exception as exc:                           # pragma: no cover
        required["jpeg2000"] = f"FAIL {exc}"

    try:
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "probe.txt").write_text("ok", encoding="utf-8")
        required["临时目录可写"] = "ok（单文件 exe 需要向 %TEMP% 解包）"
    except Exception as exc:                           # pragma: no cover
        required["临时目录可写"] = f"FAIL {exc}"

    failed = [k for k, v in required.items() if str(v).startswith("FAIL")]
    for key, value in required.items():
        LOG.info("自检[必需] %-14s %s", key, value)
    for key, value in optional.items():
        LOG.info("自检[可选] %-14s %s", key, value)
    LOG.info("自检结论：%s", "全部通过" if not failed else f"存在失败项 {failed}")

    if report:
        Path(report).parent.mkdir(parents=True, exist_ok=True)
        Path(report).write_text(
            json.dumps(
                {
                    "required": required,
                    "optional": optional,
                    "failed": failed,
                    "verdict": "pass" if not failed else "fail",
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        LOG.info("自检报告：%s", report)
    return 0 if not failed else 2


def _to_dict(result: Result) -> dict:
    return {
        "source": result.src,
        "output": result.dst,
        "page_count": result.page_count,
        "elapsed_sec": round(result.elapsed, 2),
        "output_bytes": result.output_bytes,
        "input_bytes": os.path.getsize(result.src) if os.path.exists(result.src) else None,
        "pages": [
            {
                "page": p.index + 1,
                "width": p.width,
                "height": p.height,
                "payload_bytes": p.payload,
                "white_point": round(p.info.get("Wp", 0.0), 2),
                "paper_level": round(p.info.get("paper_level", 0.0), 2),
                "mode": p.info.get("mode", ""),
                "holes_filled": p.info.get("filled", 0),
                "holes_skipped": p.info.get("skipped", 0),
            }
            for p in result.pages
        ],
        "validation": (
            [
                {
                    "page": v.page,
                    "psnr": round(v.psnr, 3),
                    "maxerr": v.maxerr,
                    "margin_dirty": round(v.margin_dirty, 6),
                }
                for v in (result.validation or [])
            ]
            if result.validation
            else None
        ),
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ClassicsClean",
        description="古籍扫描件白底清洁工具：纸面拉白，笔画不动。",
    )
    parser.add_argument("inputs", nargs="*", help="待处理的 PDF（留空则启动图形界面）")
    parser.add_argument("-o", "--output", help="输出路径；批量处理时视为目录")
    parser.add_argument("--db", type=float, default=None,
                        help=f"JPEG2000 编码质量(dB)，默认 {DEFAULT_PARAMS.jp2_db}")
    parser.add_argument("--validate", action="store_true", help="出片后回读校验（耗时约翻倍）")
    parser.add_argument("--report", help="把结果写成 JSON 报告文件")
    parser.add_argument("--selftest", action="store_true",
                        help="只做环境自检（库/JPEG2000/临时目录），不处理文件")
    parser.add_argument("--log", help="日志文件路径（GUI 子系统下默认自动生成）")
    parser.add_argument("-q", "--quiet", action="store_true", help="不打印进度")
    parser.add_argument("--version", action="version", version=f"ClassicsClean {VERSION}")
    args = parser.parse_args(argv)

    if args.selftest:
        _setup_logging(args.log, quiet=False)
        return _selftest(args.report)

    if not args.inputs:
        from .gui import launch

        return launch()

    _setup_logging(args.log, args.quiet)
    params = DEFAULT_PARAMS
    if args.db:
        params = params.with_db(args.db)

    outputs = []
    exit_code = 0
    for src in args.inputs:
        src = str(Path(src))
        if not os.path.isfile(src):
            LOG.error("文件不存在：%s", src)
            exit_code = 1
            continue
        if args.output:
            out = args.output
            if len(args.inputs) > 1 or os.path.isdir(args.output):
                out = str(Path(args.output) / f"{Path(src).stem}_清洁版.pdf")
        else:
            out = _default_output(src)
        LOG.info("开始处理：%s -> %s", src, out)
        t0 = time.time()
        try:
            result = process_pdf(src, out, params, progress=_progress,
                                 validate=args.validate or None)
        except Exception as exc:              # 单文件失败不影响批量其余文件
            LOG.exception("处理失败：%s (%s)", src, exc)
            exit_code = 1
            continue
        mb_out = result.output_bytes / 1048576
        mb_in = os.path.getsize(src) / 1048576
        LOG.info("完成：%d 页，%.2f MB -> %.2f MB，用时 %.1fs，输出：%s",
                 result.page_count, mb_in, mb_out, time.time() - t0, result.dst)
        if result.validation:
            worst = min(v.psnr for v in result.validation)
            LOG.info("回读校验：最差 PSNR %.2f dB，最大像素误差 %d",
                     worst, max(v.maxerr for v in result.validation))
        outputs.append(result)

    if args.report and outputs:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(
            json.dumps({"results": [_to_dict(r) for r in outputs]}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        LOG.info("报告已写入：%s", args.report)
    return exit_code
