# -*- coding: utf-8 -*-
"""morphology 纯 numpy 算子与 cv2 的逐像素对照测试。

morphology.py 的存在意义就是「不装 OpenCV 也能得到逐像素相同的结果」，
本文件把这个承诺固化成断言：每个算子都在同一份随机数据上与 cv2 对照。

cv2 未安装时全部跳过（日常运行与 CI 打包机都不需要 cv2）；
开发机装了 cv2 时，这里是算法迁移的回归保险。
"""
from __future__ import annotations

import numpy as np
import pytest

from classicsclean import morphology as mp

cv2 = pytest.importorskip("cv2", reason="cv2 未安装：跳过纯 numpy 与 OpenCV 的对照测试")

RADIUS_APP = (2, 3, 8, 14)
"""repair.py 实际用到的圆盘半径（含 rim+6=14）。"""


@pytest.fixture(scope="module")
def data():
    """稠密随机掩膜 + 灰度图 + 贴边掩膜，覆盖各种边界条件。"""
    rng = np.random.default_rng(20261017)
    mask = rng.random((53, 47)) > 0.72
    mask[0:3, 0:3] = True          # 触及图像边界，检验越界处理
    mask[-2:, -2:] = True
    gray = (rng.random((53, 47)) * 255).astype(np.uint8)
    float32 = (rng.random((53, 47)) * 255).astype(np.float32)
    sparse = np.zeros((61, 43), bool)   # 稀疏且包围盒不贴满全图
    sparse[10:14, 8:12] = True
    sparse[40, 30] = True
    sparse[0, 0] = True
    return mask, gray, float32, sparse


# ---------------------------------------------------------------- 结构元
@pytest.mark.parametrize("r", list(range(0, 25)))
def test_disk_matches_cv2_ellipse(r):
    """disk(r) 必须逐像素等于 cv2.MORPH_ELLIPSE((2r+1, 2r+1))。

    OpenCV 的椭圆核按行取半宽 dx = cvRound(r*sqrt(1-(dy/r)^2))，
    四舍五入而非截断——这是最容易踩空的一步。
    """
    k_cv = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    assert np.array_equal(k_cv, mp.disk(r).astype(np.uint8))


def test_disk_rejects_negative_radius():
    with pytest.raises(ValueError):
        mp.disk(-1)


# ---------------------------------------------------------------- 二值形态学
@pytest.mark.parametrize("r", (1, 2, 3, 8, 14, 21))
@pytest.mark.parametrize("which", ("dense", "sparse"))
def test_dilate_disk_matches_cv2(data, r, which):
    mask = data[0] if which == "dense" else data[3]
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    ref = cv2.dilate(mask.astype(np.uint8), k) > 0
    assert np.array_equal(ref, mp.dilate(mask, mp.disk(r)))


@pytest.mark.parametrize("r", (1, 2, 3, 8, 14, 21))
@pytest.mark.parametrize("which", ("dense", "sparse"))
def test_erode_disk_matches_cv2(data, r, which):
    mask = data[0] if which == "dense" else data[3]
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    ref = cv2.erode(mask.astype(np.uint8), k) > 0
    assert np.array_equal(ref, mp.erode(mask, mp.disk(r)))


@pytest.mark.parametrize("r", RADIUS_APP)
@pytest.mark.parametrize("which", ("dense", "sparse"))
def test_closing_disk_matches_cv2(data, r, which):
    """闭运算 = 先膨胀后腐蚀，必须与 cv2.morphologyEx(MORPH_CLOSE) 一致。"""
    mask = data[0] if which == "dense" else data[3]
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    ref = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, k) > 0
    assert np.array_equal(ref, mp.closing(mask, mp.disk(r)))


@pytest.mark.parametrize("n", (1, 2, 3, 8, 16))
@pytest.mark.parametrize("which", ("dense", "sparse"))
def test_dilate_square_iterations_matches_cv2(data, n, which):
    """3x3 方形核迭代 n 次 == square(n) 一次（极大值运算可结合）。"""
    mask = data[0] if which == "dense" else data[3]
    ref = cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=n) > 0
    assert np.array_equal(ref, mp.dilate(mask, mp.square(n)))


# ---------------------------------------------------------------- 线性 / 非线性滤波
@pytest.mark.parametrize("n", (1, 3, 12))
def test_max_filter_matches_cv2_dilate(data, n):
    """max_filter == cv2.dilate(3x3, iterations=n)（灰度极大值，越界按 0 计）。"""
    gray = data[1]
    ref = cv2.dilate(gray, np.ones((3, 3), np.uint8), iterations=n)
    assert np.array_equal(ref, mp.max_filter(gray, n))


@pytest.mark.parametrize("w", (3, 7, 31, 61))
def test_box_blur_matches_cv2(data, w):
    """box_blur == cv2.blur(BORDER_REFLECT)，允许 float32 累加误差。"""
    a = data[2]
    ref = cv2.blur(a, (w, w), borderType=cv2.BORDER_REFLECT)
    assert np.abs(ref.astype(np.float64) - mp.box_blur(a, w).astype(np.float64)).max() <= 1e-3


@pytest.mark.parametrize("s", (0.8, 1.5, 2.0, 3.0))
def test_gaussian_blur_matches_cv2(data, s):
    """gaussian_blur == cv2.GaussianBlur(ksize=(0,0))。

    核尺寸规则 cvRound(8*sigma+1)|1、核函数 exp(-x^2/(2*sigma^2))、
    边界 BORDER_REFLECT_101，缺一不可。
    """
    a = data[2]
    ref = cv2.GaussianBlur(a, (0, 0), s)
    assert np.abs(ref.astype(np.float64) - mp.gaussian_blur(a, s).astype(np.float64)).max() <= 1e-3


@pytest.mark.parametrize("k", (3, 5))
def test_median_blur_matches_cv2(data, k):
    gray = data[1]
    assert np.array_equal(cv2.medianBlur(gray, k), mp.median_blur(gray, k))


def test_median_blur_rejects_even_kernel(data):
    with pytest.raises(ValueError):
        mp.median_blur(data[1], 4)


# ---------------------------------------------------------------- 连通域 / 漫水
@pytest.mark.parametrize("conn", (4, 8))
def test_connected_components_matches_cv2(data, conn):
    """分量集合与面积序列必须与 cv2.connectedComponentsWithStats 一致。"""
    mask = data[0]
    ref_n, ref_lab, ref_stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=conn
    )
    comp = mp.connected_components(mask, connectivity=conn)
    assert comp.count == ref_n - 1
    # 逐个 cv2 分量核对：numpy 里同编号集合必须完全相同
    for i in range(1, ref_n):
        sel = ref_lab == i
        j = comp.labels[sel][0]
        assert np.array_equal(comp.labels == j, sel), f"分量 {i} 集合不一致"
    assert sorted(int(v) for v in ref_stats[1:, cv2.CC_STAT_AREA]) == sorted(
        int(v) for v in comp.areas[1:]
    )


def test_region_from_seeds_matches_flood_fill(data):
    """全部种子并集 == 逐种子 cv2.floodFill（默认 4 连通）的并集。"""
    mask = data[0]
    bright = mask.astype(np.uint8)
    h, w = bright.shape
    fmask = np.zeros((h + 2, w + 2), np.uint8)
    ys, xs = np.nonzero(bright)
    for y, x in list(zip(ys, xs))[:: max(1, len(ys) // 2000)]:
        cv2.floodFill(bright, fmask, (int(x), int(y)), 2)
    seed = np.zeros_like(mask)
    seed[ys, xs] = True
    assert np.array_equal(bright == 2, mp.region_from_seeds(mask, seed, connectivity=4))


def test_connected_components_empty_mask():
    comp = mp.connected_components(np.zeros((5, 5), bool))
    assert comp.count == 0
    assert comp.labels.sum() == 0


# ---------------------------------------------------------------- 面积缩放
@pytest.mark.parametrize(
    "dw,dh",
    [(40, 50), (48, 24), (97, 41), (166, 194), (24, 97), (83, 49), (9, 7), (31, 31), (17, 13)],
)
def test_area_resize_matches_cv2(data, dw, dh):
    """area_resize 复刻 cv2.INTER_AREA：整数倍走块和快路径，其余走权重表。

    float64 累加 vs cv2 的 float32 累加，只允许四舍五入口径上的 ±1 灰阶。
    """
    base = (np.random.default_rng(7).random((97, 83)) * 255).astype(np.uint8)
    ref = cv2.resize(base, (dw, dh), interpolation=cv2.INTER_AREA)
    got = mp.area_resize(base, dw, dh)
    assert np.abs(ref.astype(np.int32) - got.astype(np.int32)).max() <= 1


def test_area_resize_identity_is_copy(data):
    a = data[1]
    out = mp.area_resize(a, a.shape[1], a.shape[0])
    assert np.array_equal(out, a)
    assert out is not a


def test_area_resize_rejects_bad_shape():
    with pytest.raises(ValueError):
        mp.area_resize(np.zeros((4, 4), np.uint8), 0, 4)
    with pytest.raises(ValueError):
        mp.area_resize(np.zeros((4, 4, 3), np.uint8), 2, 2)
