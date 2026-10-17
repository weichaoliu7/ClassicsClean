# -*- coding: utf-8 -*-
"""配置单元测试：参数唯一来源必须自洽，非法参数必须在构造时就报错。"""
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

import classicsclean
from classicsclean.config import (
    APP_NAME,
    PERCENTILE_MODES,
    VERSION,
    WHITE_POINT_MODES,
    DEFAULT_PARAMS,
    Params,
)


def test_version_matches_pyproject():
    """包内版本号必须与 pyproject.toml 一致（避免两处漂移）。"""
    import tomllib

    data = tomllib.loads((Path(__file__).resolve().parents[2] / "pyproject.toml")
                         .read_text(encoding="utf-8"))
    assert data["project"]["version"] == VERSION


def test_public_api_exports():
    """包顶层必须导出文档里承诺的那几个名字。"""
    for name in ("process_pdf", "validate_output", "level_page", "Params", "DEFAULT_PARAMS"):
        assert hasattr(classicsclean, name), f"缺少公开 API：{name}"
    assert classicsclean.__version__ == VERSION
    assert APP_NAME == "ClassicsClean"


def test_default_params_are_the_documented_values():
    """默认值必须与 README / docs 中记录的一致（避免文档漂移）。"""
    assert DEFAULT_PARAMS.wp_mode == "local"
    assert DEFAULT_PARAMS.gamma == 3.0
    assert DEFAULT_PARAMS.jp2_db == 52.0
    assert DEFAULT_PARAMS.text_guard == 1
    assert DEFAULT_PARAMS.repair_holes_enabled is True
    assert DEFAULT_PARAMS.validate is False


def test_params_is_frozen():
    """参数不可变：任何就地修改都必须抛异常。"""
    with pytest.raises(FrozenInstanceError):
        DEFAULT_PARAMS.gamma = 2.0  # type: ignore[misc]


def test_curve_selection():
    """gamma=1 走线性，其余走 gamma 高光压缩。"""
    assert Params(gamma=1.0).curve() == ("linear",)
    assert Params(gamma=3.0).curve()[0] == "gamma"
    assert Params(gamma=3.0).curve()[1] == 3.0


def test_with_db_returns_copy():
    """with_db 必须返回新对象且不改动原对象。"""
    other = DEFAULT_PARAMS.with_db(58.0)
    assert other is not DEFAULT_PARAMS
    assert other.jp2_db == 58.0
    assert DEFAULT_PARAMS.jp2_db == 52.0
    assert other.gamma == DEFAULT_PARAMS.gamma


@pytest.mark.parametrize(
    "kwargs",
    [
        {"gamma": 0.0},
        {"gamma": -1.0},
        {"wp_mode": "bogus"},
        {"min_area_frac": 0.0},
        {"min_area_frac": 1.5},
        {"jp2_db": 200.0},
        {"ink_ratio": 1.5},
    ],
)
def test_invalid_params_rejected(kwargs):
    """非法参数必须在构造时就被拒绝，而不是跑完一本书才发现。"""
    with pytest.raises(ValueError):
        Params(**kwargs)


def test_white_point_modes_cover_percentiles():
    """分位模式必须被纳入合法白场模式集合。"""
    for mode in PERCENTILE_MODES:
        assert mode in WHITE_POINT_MODES
    assert Params(wp_mode="p65").wp_mode == "p65"
