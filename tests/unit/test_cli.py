# -*- coding: utf-8 -*-
"""命令行单元测试：批处理出口必须能跑通、能报错、能出报告。"""
import json
import os
from pathlib import Path

import pytest

import classicsclean
from classicsclean.apps.cli import main


def test_version_flag_exits_zero(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert classicsclean.VERSION in capsys.readouterr().out


def test_selftest_passes_and_writes_report(tmp_path):
    """环境自检必须给出明确结论，并写出机器可读报告。"""
    report = tmp_path / "selftest.json"
    assert main(["--selftest", "--report", str(report)]) == 0
    data = json.loads(report.read_text(encoding="utf-8"))
    assert data["verdict"] == "pass" and data["failed"] == []
    for key in ("numpy", "pymupdf", "pikepdf", "PIL", "jpeg2000", "临时目录可写"):
        assert data["required"][key].startswith("ok"), f"{key} 自检失败：{data['required'][key]}"
    # cv2 已于 v1.2 移除（纯 numpy morphology 取代），不应再出现在必需项里
    assert "cv2" not in data["required"]
    assert "tkinter" in data["optional"]


def test_batch_conversion_and_report(tmp_path, scan_pdf):
    """一次批量处理：出片 + 写 JSON 报告。"""
    out = tmp_path / "out.pdf"
    report = tmp_path / "report.json"
    assert main([scan_pdf, "-o", str(out), "--report", str(report)]) == 0
    assert out.is_file() and out.stat().st_size > 0

    data = json.loads(report.read_text(encoding="utf-8"))
    result = data["results"][0]
    assert result["page_count"] == 3
    assert result["output_bytes"] == os.path.getsize(str(out))
    assert len(result["pages"]) == 3
    assert result["validation"] is None


def test_validate_flag_adds_rows(tmp_path, scan_pdf):
    """--validate 必须带回逐页回读校验数据。"""
    out = tmp_path / "out.pdf"
    report = tmp_path / "report.json"
    assert main([scan_pdf, "-o", str(out), "--validate", "--report", str(report)]) == 0
    data = json.loads(report.read_text(encoding="utf-8"))
    rows = data["results"][0]["validation"]
    assert rows and len(rows) == 3
    assert all(r["psnr"] > 35.0 for r in rows)


def test_custom_db_is_honoured(tmp_path, scan_pdf):
    """--db 必须真实影响编码质量（高 dB -> 更大体积）。"""
    low = tmp_path / "low.pdf"
    high = tmp_path / "high.pdf"
    main([scan_pdf, "-o", str(low), "--db", "40"])
    main([scan_pdf, "-o", str(high), "--db", "62"])
    assert high.stat().st_size > low.stat().st_size


def test_missing_file_reports_error(tmp_path, caplog):
    """输入不存在时返回非 0，且不影响其它文件（批量容错）。"""
    assert main([str(tmp_path / "nope.pdf")]) == 1


def test_default_output_name_is_next_to_source(tmp_path, scan_pdf):
    """不带 -o 时输出到源文件同目录，文件名追加 _清洁版。"""
    src = Path(scan_pdf)
    assert main([scan_pdf]) == 0
    expected = src.with_name(f"{src.stem}_清洁版.pdf")
    assert expected.is_file()
    expected.unlink()
