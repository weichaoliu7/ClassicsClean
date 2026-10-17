# -*- coding: utf-8 -*-
"""【第 1 层 · 基础】形态学、局部滤波与重采样原语（纯 numpy，无 OpenCV / SciPy）。

为什么不用 OpenCV：本项目的算法只用到几个算子（膨胀、闭运算、均值/高斯/中值
滤波、连通域标记、漫水填充）加一个重采样（面积插值缩放），而 opencv-python 的
二进制体积超过 110 MB，占打包产物的一半以上。实测表明改用 scipy.ndimage 也无效——
scipy.ndimage 会连带导入 special / linalg / sparse，打包体积与 cv2 持平（两者均约
63 MB）。因此这里用 numpy 自行实现，全部算子均为 O(1)/像素的可分离算法：

    方形结构元膨胀/腐蚀   —— 滑动窗口求和（前缀和，O(1)/像素）
    圆盘结构元膨胀/腐蚀   —— 逐行滑动窗口求和（O(半径)/像素，且在包围盒内计算）
    灰度极大值滤波        —— van Herk / Gil-Werman 算法（O(1)/像素）
    均值滤波              —— 前缀和（float64 中间结果，避免长序列累积误差）
    高斯滤波              —— 可分离卷积，核尺寸与 OpenCV 一致
    中值滤波              —— 滑动窗口视图 + 分块求中位数（限制峰值内存）
    连通域标记            —— 逐行游程 + 并查集（O(游程数)）
    面积插值缩放          —— DecimateAlpha 权重表（与 OpenCV INTER_AREA 同式）

**与 cv2 的等价性**（tests/unit/test_morphology.py 逐项对照，cv2 未安装时自动跳过；
日常运行不依赖 cv2）。三处容易踩坑的细节，改动前必读：

1. `disk()` 不是几何圆盘，而是逐像素复刻 OpenCV `MORPH_ELLIPSE`：每行半宽
   `dx = cvRound(r*sqrt(1-(dy/r)^2))`（`cvRound` 是四舍五入，不是截断！）。
   直接相减会得到另一块内核，膨胀结果整页都会错。
2. `gaussian_blur()` 的核尺寸规则是 `cvRound(8*sigma+1)|1`（浮点输入；
   uint8 输入为 `cvRound(6*sigma+1)|1`），核函数为 `exp(-x^2/(2*sigma^2))`
   归一化。用 `round(3*sigma)` 当半径会得到更窄的核，羽化范围整体偏窄。
3. `area_resize()` 完整复刻 `cv2.INTER_AREA`：整数倍缩放走「整数块和 / 块大小」
   快路径，其余走 DecimateAlpha 权重表；放大（scale<1）时 OpenCV 退化为带
   面积采样位置的 bilinear，此处同款实现。

边界：只提供**无状态**的数组算子，不含任何古籍处理领域的语义。
"""
from __future__ import annotations

import math
from typing import Callable, List, NamedTuple, Optional, Tuple

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

__all__ = [
    "disk",
    "square",
    "dilate",
    "erode",
    "closing",
    "max_filter",
    "box_blur",
    "gaussian_blur",
    "gaussian_radius",
    "median_blur",
    "area_resize",
    "bbox_of",
    "Components",
    "connected_components",
    "region_from_seeds",
]


# ============================================================================
# 结构元
# ============================================================================
def square(radius: int) -> np.ndarray:
    """半径 radius 的实心方形结构元（边长 2r+1）。"""
    r = int(radius)
    if r < 0:
        raise ValueError(f"radius 不能为负，当前 {radius}")
    return np.ones((2 * r + 1, 2 * r + 1), dtype=bool)


def disk(radius: int) -> np.ndarray:
    """半径 radius 的圆盘结构元（逐像素复刻 cv2.MORPH_ELLIPSE 的方形核）。

    OpenCV 的 MORPH_ELLIPSE 不是几何圆盘，而是按行计算半宽：

        dx = cvRound(c * sqrt((r*r - dy*dy) / (r*r)))     # c == r（方形核）

    即每行保留 |x| <= dx 的连续一段，dx 四舍五入（cvRound）而非截断。
    本项目全部调用点（repair.py 的破洞修补）使用的都是 cv2.MORPH_ELLIPSE 核，
    形状必须逐像素一致，否则膨胀范围整体偏移。
    """
    r = int(radius)
    if r < 0:
        raise ValueError(f"radius 不能为负，当前 {radius}")
    if r == 0:
        return np.ones((1, 1), dtype=bool)          # cv2: ksize=1x1 退化为矩形
    inv_r2 = 1.0 / (r * r)
    out = np.zeros((2 * r + 1, 2 * r + 1), dtype=bool)
    for i in range(2 * r + 1):
        dy = i - r
        dx = int(r * math.sqrt((r * r - dy * dy) * inv_r2) + 0.5)   # cvRound
        out[i, r - dx : r + dx + 1] = True
    return out


# ============================================================================
# 基础工具
# ============================================================================
def _as_mask(mask: np.ndarray) -> np.ndarray:
    arr = np.asarray(mask)
    if arr.dtype != np.bool_:
        arr = arr.astype(bool)
    return arr


def _box_sum(arr: np.ndarray, win: int, axis: int, dtype=np.int64) -> np.ndarray:
    """沿 axis 的滑动窗口求和，输出尺寸与输入相同（越界按 0 计）。"""
    n = arr.shape[axis]
    if axis == 0:
        cum = np.zeros((n + 1, arr.shape[1]), dtype=dtype)
        np.cumsum(arr, axis=0, out=cum[1:], dtype=dtype)
        return cum[win:, :] - cum[:-win, :]
    cum = np.zeros((arr.shape[0], n + 1), dtype=dtype)
    np.cumsum(arr, axis=1, out=cum[:, 1:], dtype=dtype)
    return cum[:, win:] - cum[:, :-win]


def bbox_of(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """非零区域的包围盒 (r0, r1, c0, c1)（右/下开区间），全空返回 None。"""
    rows = np.flatnonzero(mask.any(axis=1))
    if rows.size == 0:
        return None
    cols = np.flatnonzero(mask.any(axis=0))
    return int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1


_bbox_of = bbox_of  # 内部别名（_morph 使用）


def _shift_vertical(mask: np.ndarray, offset: int, pad: bool) -> np.ndarray:
    """纵向平移 offset 行（正=向下），空出部分填 pad。"""
    out = np.full(mask.shape, pad, dtype=bool)
    h = mask.shape[0]
    if offset >= 0:
        if offset < h:
            out[offset:, :] = mask[: h - offset, :]
    elif -offset < h:
        out[: h + offset, :] = mask[-offset:, :]
    return out


def _sliding_any(arr: np.ndarray, left: int, right: int, pad: bool) -> np.ndarray:
    """横向滑动窗口「是否存在 True」：out[:, x] = any(arr[:, x-left .. x+right])。

    left / right 必须非负，越界部分按 pad 计（pad=True 视为 True）。
    """
    if left < 0 or right < 0:
        raise ValueError(f"窗口延展必须非负，当前 left={left}, right={right}")
    value = 1 if pad else 0
    work = np.pad(arr.astype(np.int32), ((0, 0), (left, right)), constant_values=value)
    total = _box_sum(work, left + right + 1, axis=1, dtype=np.int32)
    return total >= (left + right + 1) if pad else total > 0


# ============================================================================
# 二值形态学
# ============================================================================
def _morph(mask: np.ndarray, se: np.ndarray, op: str) -> np.ndarray:
    """二值膨胀/腐蚀：op ∈ {"dilate", "erode"}。

    膨胀：核内存在 True 即 True（越界视为 False）
    腐蚀：核内全为 True 才 True（越界视为 True，等价于 OpenCV 忽略越界像素）
    """
    m = _as_mask(mask)
    se = np.asarray(se, dtype=bool)
    if se.ndim != 2 or se.shape[0] % 2 == 0 or se.shape[1] % 2 == 0:
        raise ValueError(f"结构元必须是奇数边长的二维数组，当前 {se.shape}")
    want = op == "dilate"
    pad = not want  # 腐蚀时越界补 True（不影响结果），膨胀时补 False

    ch, cw = se.shape[0] // 2, se.shape[1] // 2
    rows = np.flatnonzero(se.any(axis=1))
    cols_of = [np.flatnonzero(se[u]) for u in rows]
    if not any(c.size for c in cols_of):
        return np.zeros_like(m)

    # 矩形结构元（含默认 3×3）-> 两趟可分离滑动窗口，与核半径无关
    rectangular = bool(np.all(se))
    pad_value = 1 if pad else 0
    if rectangular:
        work = np.pad(m.astype(np.int32), ((ch, ch), (cw, cw)), constant_values=pad_value)
        total = _box_sum(
            _box_sum(work, se.shape[0], 0, np.int32), se.shape[1], 1, np.int32
        )
        return total >= se.size if pad else total > 0

    # 一般（圆盘）结构元：在包围盒内逐行处理
    box = _bbox_of(m)
    if box is None:
        return np.zeros_like(m)
    br0, br1, bc0, bc1 = box                       # 原始包围盒（右/下开区间）
    r0 = max(br0 - ch, 0)                          # 计算用外扩盒（ clamped 到图内 ）
    r1 = min(br1 + ch, m.shape[0])
    c0 = max(bc0 - cw, 0)
    c1 = min(bc1 + cw, m.shape[1])
    sub = m[r0:r1, c0:c1]

    # 累加器取对应逻辑运算的单位元：膨胀从 False 起 OR，腐蚀从 True 起 AND
    acc = np.full(sub.shape, not want, dtype=bool)
    for u, cols in zip(rows, cols_of):
        if cols.size == 0:
            continue
        a = int(cols[0]) - cw
        b = int(cols[-1]) - cw
        if not np.array_equal(cols, np.arange(cols[0], cols[-1] + 1)):
            raise ValueError("结构元的每一行必须是连续的一段")
        # out[y, x] = OR/AND over dx in [a, b] of sub[y - dy, x - dx]
        src = _shift_vertical(sub, int(u) - ch, pad)
        hit = _sliding_any(src, b, -a, pad)
        acc = np.logical_or(acc, hit) if want else np.logical_and(acc, hit)

    # 膨胀结果铺满外扩盒；腐蚀结果只可能在原始包围盒内（结构元必含中心，
    # 盒外像素的核中心落在掩膜外，恒为 False），外扩盒里多算的一圈必须丢掉
    out = np.zeros(m.shape, dtype=bool)
    if want:
        out[r0:r1, c0:c1] = acc
    else:
        out[br0:br1, bc0:bc1] = acc[br0 - r0 : br1 - r0, bc0 - c0 : bc1 - c0]
    return out


def dilate(mask: np.ndarray, se: Optional[np.ndarray] = None) -> np.ndarray:
    """二值膨胀。默认 3×3 方形核（等价 cv2.dilate 的默认核）。

    注意：3×3 方形核迭代 n 次 == 半径 n 的方形核一次（极大值运算可结合），
    因此调用方直接传 square(n) 即可，无需 iterations 参数。
    """
    return _morph(mask, square(1) if se is None else se, "dilate")


def erode(mask: np.ndarray, se: Optional[np.ndarray] = None) -> np.ndarray:
    """二值腐蚀。默认 3×3 方形核。"""
    return _morph(mask, square(1) if se is None else se, "erode")


def closing(mask: np.ndarray, se: Optional[np.ndarray] = None) -> np.ndarray:
    """二值闭运算 = 先膨胀后腐蚀（等价 cv2.MORPH_CLOSE）。"""
    if se is None:
        se = square(1)
    return erode(dilate(mask, se), se)


def max_filter(arr: np.ndarray, radius: int) -> np.ndarray:
    """灰度极大值滤波（方形核，半径 radius）。

    等价于 cv2.dilate(a, np.ones((3,3)), iterations=radius)，
    但用 van Herk / Gil-Werman 算法实现，复杂度与半径无关。
    越界像素按 0 计（本项目像素值恒 >= 0，等价于 OpenCV 忽略越界）。
    """
    a = np.asarray(arr)
    r = int(radius)
    if r < 0:
        raise ValueError(f"radius 不能为负，当前 {radius}")
    if r == 0:
        return a.copy()
    size = 2 * r + 1
    out = _sliding_max(a, size, axis=0)
    return _sliding_max(out, size, axis=1)


def _sliding_max(arr: np.ndarray, size: int, axis: int) -> np.ndarray:
    """out[i] = max(arr[i-r .. i+r])，越界按 0 计；要求 arr >= 0。"""
    r = size // 2
    work = np.moveaxis(np.asarray(arr), axis, -1)
    n = work.shape[-1]
    total = n + 2 * r
    blocks = -(-total // size)  # ceil
    width = blocks * size
    buf = np.zeros(work.shape[:-1] + (width,), dtype=work.dtype)
    buf[..., r : r + n] = work
    flat = buf.reshape(-1, blocks, size)
    fwd = np.maximum.accumulate(flat, axis=-1)
    bwd = np.maximum.accumulate(flat[..., ::-1], axis=-1)[..., ::-1]
    fwd = fwd.reshape(buf.shape)
    bwd = bwd.reshape(buf.shape)
    res = np.maximum(fwd[..., 2 * r : 2 * r + n], bwd[..., :n])
    return np.moveaxis(res, -1, axis)


# ============================================================================
# 线性滤波
# ============================================================================
def box_blur(arr: np.ndarray, win: int) -> np.ndarray:
    """win×win 均值滤波，边界 BORDER_REFLECT（等价 cv2.blur）。

    中间结果用 float64，避免 float32 前缀和在长行/长列上的累积误差。
    """
    a = np.asarray(arr)
    w = int(win)
    if w < 1 or w % 2 == 0:
        raise ValueError(f"win 必须为正奇数，当前 {win}")
    q = w // 2
    work = np.pad(a.astype(np.float64), q, mode="symmetric")
    total = _box_sum(_box_sum(work, w, 0, np.float64), w, 1, np.float64)
    out = total / float(w * w)
    return out.astype(np.float64 if a.dtype == np.float64 else np.float32)


def _gaussian_taps(sigma: float, is_uint8: bool = False) -> np.ndarray:
    """OpenCV 在 ksize=(0,0) 时使用的归一化高斯核。

    核尺寸规则（OpenCV createGaussianKernels，按输入深度分档）：

        浮点输入:  n = cvRound(8*sigma + 1) | 1      # 半径约 4σ
        uint8 输入: n = cvRound(6*sigma + 1) | 1      # 半径约 3σ

    核函数为 exp(-x^2/(2*sigma^2)) 归一化，x = i - (n-1)/2。
    注意不是 round(3*sigma)：那会得到更窄的核，文字保护区外圈的羽化宽度
    会整体偏小，与 cv2 版本逐像素对不上。
    """
    s = float(sigma)
    span = 6.0 if is_uint8 else 8.0
    n = int(span * s + 1.0 + 0.5) | 1               # cvRound(span*sigma+1) | 1
    n = max(n, 1)
    x = np.arange(n, dtype=np.float64) - (n - 1) / 2.0
    k = np.exp(-(x * x) / (2.0 * s * s))
    return k / k.sum()


def gaussian_radius(sigma: float) -> int:
    """sigma 对应的核半径（与 gaussian_blur 同一尺寸规则，浮点输入档）。

    供调用方裁包围盒用：高斯结果在 mask 包围盒外扩该半径后恒为 0，
    裁到那里计算与整页计算逐像素一致（reflect 边界读到的全是 0）。
    """
    n = int(8.0 * float(sigma) + 1.0 + 0.5) | 1
    return max(n, 1) // 2


def gaussian_blur(arr: np.ndarray, sigma: float) -> np.ndarray:
    """sigma 指定的可分离高斯滤波（复刻 cv2.GaussianBlur(ksize=(0,0))）。

    边界 BORDER_REFLECT_101（cv2.GaussianBlur 的默认 BORDER_DEFAULT）；
    先横向后纵向（与 OpenCV 的 row/column 滤波顺序一致）。

    适用范围：float32 / float64 输入（本项目两处调用均为 float32）。
    cv2 对 uint8 输入走定点 bit-exact 路径，这里不模拟，请勿传入 uint8。
    """
    a = np.asarray(arr)
    s = float(sigma)
    if s <= 0:
        return a.astype(np.float64 if a.dtype == np.float64 else np.float32, copy=True)
    taps = _gaussian_taps(s, is_uint8=(a.dtype == np.uint8))
    r = len(taps) // 2
    h, w = a.shape[0], a.shape[1]

    # 横向：只在宽度方向反射补齐，纵向同理，避免两轴同时补后边界语义偏移
    work = np.pad(a.astype(np.float64), ((0, 0), (r, r)), mode="reflect")
    tmp = np.zeros((h, w), dtype=np.float64)
    for t, weight in enumerate(taps):
        tmp += weight * work[:, t : t + w]
    work = np.pad(tmp, ((r, r), (0, 0)), mode="reflect")
    out = np.zeros((h, w), dtype=np.float64)
    for t, weight in enumerate(taps):
        out += weight * work[t : t + h, :]
    return out.astype(np.float64 if a.dtype == np.float64 else np.float32)


def median_blur(arr: np.ndarray, k: int) -> np.ndarray:
    """k×k 中值滤波（等价 cv2.medianBlur，边界 BORDER_REPLICATE）。

    分块计算，避免一次性构造 (H, W, k, k) 的巨大临时数组。
    """
    a = np.asarray(arr)
    k = int(k)
    if k < 3 or k % 2 == 0:
        raise ValueError(f"k 必须为 >= 3 的奇数，当前 {k}")
    q = k // 2
    padded = np.pad(a, q, mode="edge")
    out = np.empty_like(a)
    step = max(1, int(32_000_000 // max(1, k * k * max(1, a.shape[1]))))
    for r0 in range(0, a.shape[0], step):
        r1 = min(r0 + step, a.shape[0])
        win = sliding_window_view(padded[r0 : r1 + 2 * q], (k, k))
        out[r0:r1] = np.median(win, axis=(-2, -1)).astype(a.dtype)
    return out


# ============================================================================
# 重采样（复刻 cv2.resize 的 INTER_AREA）
# ============================================================================
def _decimate_weights(ssize: int, dsize: int, scale: float):
    """OpenCV computeResizeAreaTab 的权重表（scale >= 1 的下采样路径）。

    对每个输出下标 dx，源区间为 [dx*scale, (dx+1)*scale)，落在区间内的源像素
    按重叠长度加权（整段权重 1/scale，两端不足一格的部分按比例），超出图像
    右边界时截断。由于 scale = ssize/dsize，cellWidth 恒等于 scale。

    返回 (si, di, alpha)：三个等长数组，累加和即 buf[di] += src[si]*alpha。
    """
    si_l, di_l, al_l = [], [], []
    for dx in range(dsize):
        fsx1 = dx * scale
        fsx2 = fsx1 + scale
        sx1 = math.ceil(fsx1)
        sx2 = math.floor(fsx2)
        sx2 = min(sx2, ssize - 1)
        sx1 = min(sx1, sx2)
        if sx1 - fsx1 > 1e-3:
            si_l.append(sx1 - 1)
            di_l.append(dx)
            al_l.append((sx1 - fsx1) / scale)
        for sx in range(sx1, sx2):
            si_l.append(sx)
            di_l.append(dx)
            al_l.append(1.0 / scale)
        if fsx2 - sx2 > 1e-3:
            si_l.append(sx2)
            di_l.append(dx)
            al_l.append(min(min(fsx2 - sx2, 1.0), scale) / scale)
    return (
        np.asarray(si_l, dtype=np.int64),
        np.asarray(di_l, dtype=np.int64),
        np.asarray(al_l, dtype=np.float64),
    )


def _area_down(a: np.ndarray, dw: int, dh: int, scale_x: float, scale_y: float) -> np.ndarray:
    """INTER_AREA 下采样（scale_x >= 1 且 scale_y >= 1）。"""
    h, w = a.shape
    is_uint8 = a.dtype == np.uint8

    # 整数倍缩放：OpenCV 走 resizeAreaFast_（整数块和 / 块大小），单独复刻
    if (abs(scale_x - math.floor(scale_x)) < 1e-15
            and abs(scale_y - math.floor(scale_y)) < 1e-15):
        kx, ky = int(scale_x), int(scale_y)
        if is_uint8:
            src = a.astype(np.int64)
        else:
            src = a.astype(np.float64)
        # 先在宽度方向按 kx 分块求和，再在高度方向按 ky 分块求和
        if w % kx == 0:
            block = src.reshape(h, dw, kx).sum(axis=2)
        else:                                    # 右边界不齐：逐块裁剪
            block = np.zeros((h, dw), dtype=src.dtype)
            for dx in range(dw):
                block[:, dx] = src[:, dx * kx : min((dx + 1) * kx, w)].sum(axis=1)
        if h % ky == 0:
            acc = block.reshape(dh, ky, dw).sum(axis=1)
        else:
            acc = np.zeros((dh, dw), dtype=src.dtype)
            for dy in range(dh):
                acc[dy] = block[dy * ky : min((dy + 1) * ky, h), :].sum(axis=0)
        count = kx * ky
        if is_uint8:
            out = np.round(acc.astype(np.float64) / count)
            return np.clip(out, 0, 255).astype(np.uint8)
        return (acc / count).astype(a.dtype)

    # 一般下采样：DecimateAlpha 权重表，先横向后纵向（与 OpenCV 同序）
    si, di, al = _decimate_weights(w, dw, scale_x)
    vals = (a[:, si].astype(np.float64) * al).T          # (K, h)：按目标列累加到行
    buf = np.zeros((dw, h), dtype=np.float64)
    np.add.at(buf, di, vals)
    buf = buf.T                                          # (h, dw)
    sj, dj, be = _decimate_weights(h, dh, scale_y)
    acc = np.zeros((dh, dw), dtype=np.float64)
    np.add.at(acc, dj, buf[sj, :] * be[:, None])
    return _resize_finalize(acc, a.dtype)


def _area_bilinear(a: np.ndarray, dw: int, dh: int, scale_x: float, scale_y: float) -> np.ndarray:
    """INTER_AREA 放大（任一轴 scale < 1）：OpenCV 退化为带面积采样位置的 bilinear。

    采样位置与 INTER_LINEAR 不同：sx = floor(dx*scale)，fx 取「目标像素右边界
    映射回源坐标」的小数部分，即 (dx+1) - (sx+1)*inv_scale。
    """
    h, w = a.shape
    inv_x, inv_y = 1.0 / scale_x, 1.0 / scale_y

    def axis_params(ssize: int, dsize: int, scale: float, inv: float):
        src_idx = np.empty(dsize, dtype=np.int64)
        frac = np.zeros(dsize, dtype=np.float64)
        for d in range(dsize):
            sx = math.floor(d * scale)
            fx = (d + 1) - (sx + 1) * inv
            fx = 0.0 if fx <= 0 else fx - math.floor(fx)
            if sx < 0:
                sx, fx = 0, 0.0
            if sx >= ssize - 1:
                sx, fx = ssize - 1, 0.0
            src_idx[d] = sx
            frac[d] = fx
        return src_idx, frac

    xi, xf = axis_params(w, dw, scale_x, inv_x)
    yi, yf = axis_params(h, dh, scale_y, inv_y)

    src = a.astype(np.float64)
    x1 = np.minimum(xi + 1, w - 1)
    tmp = src[:, xi] * (1.0 - xf)[None, :] + src[:, x1] * xf[None, :]   # (h, dw)
    y1 = np.minimum(yi + 1, h - 1)
    acc = tmp[yi, :] * (1.0 - yf)[:, None] + tmp[y1, :] * yf[:, None]  # (dh, dw)
    return _resize_finalize(acc, a.dtype)


def _resize_finalize(acc: np.ndarray, dtype) -> np.ndarray:
    """累加值 -> 输出数组：uint8 四舍五入截断，浮点原样（cv2 saturate_cast）。"""
    if dtype == np.uint8:
        return np.clip(np.round(acc), 0, 255).astype(np.uint8)
    if dtype == np.float64:
        return acc.astype(np.float64)
    return acc.astype(np.float32)


def area_resize(arr: np.ndarray, width: int, height: int) -> np.ndarray:
    """复刻 cv2.resize(arr, (width, height), interpolation=cv2.INTER_AREA)。

    仅支持二维单通道数组（本项目的灰度图场景）。三条路径与 OpenCV 一致：
    整数倍下采样（块和快路径）、一般下采样（DecimateAlpha 权重表）、
    放大（面积采样位置的 bilinear）。内部以 float64 累加，与 cv2 的 float32
    累加相差约 1e-6，仅在四舍五入边界上可能差 1 个灰阶。
    """
    a = np.asarray(arr)
    if a.ndim != 2:
        raise ValueError(f"area_resize 仅支持二维数组，当前维度 {a.ndim}")
    h, w = a.shape
    dw, dh = int(width), int(height)
    if dw <= 0 or dh <= 0:
        raise ValueError(f"目标尺寸必须为正，当前 ({width}, {height})")
    if (w, h) == (dw, dh):
        return a.copy()
    scale_x, scale_y = w / dw, h / dh
    if scale_x >= 1.0 and scale_y >= 1.0:
        return _area_down(a, dw, dh, scale_x, scale_y)
    return _area_bilinear(a, dw, dh, scale_x, scale_y)


# ============================================================================
# 连通域
# ============================================================================
class Components(NamedTuple):
    """连通域标记结果（对应 cv2.connectedComponentsWithStats 的子集）。"""

    labels: np.ndarray                 # int32，与输入同尺寸，0 = 背景
    count: int                         # 前景分量个数（不含背景）
    areas: np.ndarray                  # areas[i] = 第 i 个分量的像素数，areas[0] == 0
    slices: Tuple[Tuple[slice, slice], ...]  # 每个分量的包围盒切片，索引 0 无意义

    def ids_where(self, predicate: Callable[[int], bool]) -> np.ndarray:
        """返回满足 predicate(area) 的分量编号。"""
        if self.count == 0:
            return np.empty(0, dtype=np.int32)
        keep = np.array([predicate(int(v)) for v in self.areas[1:]], dtype=bool)
        return np.flatnonzero(keep).astype(np.int32) + 1

    def mask_of(self, ids: np.ndarray) -> np.ndarray:
        """给定分量编号，返回合并后的 bool 掩膜。"""
        if np.size(ids) == 0:
            return np.zeros_like(self.labels, dtype=bool)
        return np.isin(self.labels, np.asarray(ids))


def _row_runs(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """逐行提取连续 True 段。

    返回 (row, start, stop)：stop 为开区间，三个数组等长且按行主序排列。
    """
    h, w = mask.shape
    padded = np.zeros((h, w + 2), dtype=bool)
    padded[:, 1:-1] = mask
    diff = np.diff(padded.astype(np.int8), axis=1)
    starts = np.nonzero(diff > 0)
    stops = np.nonzero(diff < 0)
    return starts[0], starts[1], stops[1]


def _union_runs(
    row: np.ndarray,
    start: np.ndarray,
    stop: np.ndarray,
    width: int,
    diagonal: bool,
) -> List[int]:
    """对相邻行的游程做并查集合并，返回 parent 数组（长度为游程数）。"""
    n = len(row)
    parent = list(range(n))
    if n == 0:
        return parent

    def find(x: int) -> int:
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if ra < rb:
            parent[rb] = ra
        else:
            parent[ra] = rb

    n_rows = int(row[-1]) + 1
    counts = np.bincount(row, minlength=n_rows)
    group = np.zeros(n_rows + 1, dtype=np.int64)
    np.cumsum(counts, out=group[1:])

    base = width + 2  # 复合键的进制，保证 row*(base) 不跨行
    key_stop = row * base + stop
    key_start = row * base + start
    shift = 0 if diagonal else 1  # 4 连通要求真正重叠而非斜接

    for r in range(1, n_rows):
        j0, j1 = int(group[r]), int(group[r + 1])
        if j1 <= j0:
            continue
        p0, p1 = int(group[r - 1]), int(group[r])
        if p1 <= p0:
            continue
        lo = np.searchsorted(key_stop, (r - 1) * base + start[j0:j1] + shift, side="left")
        hi = np.searchsorted(key_start, (r - 1) * base + stop[j0:j1] - shift, side="right") - 1
        lo = np.maximum(lo, p0)
        hi = np.minimum(hi, p1 - 1)
        for k in range(j1 - j0):
            a = int(lo[k])
            b = int(hi[k])
            if a > b:
                continue
            j = j0 + k
            union(j, a)
            for i in range(a + 1, b + 1):
                union(i, j)
    return parent


def connected_components(mask: np.ndarray, connectivity: int = 8) -> Components:
    """标记二值掩膜的连通域（默认 8 连通，与 cv2 一致）。

    算法：逐行拆成游程 -> 相邻行游程按重叠关系合并 -> 并查集求根 -> 回填。
    复杂度 O(像素数 + 游程数)，与分量形状无关。
    """
    m = _as_mask(mask)
    if not m.any():
        return Components(
            labels=np.zeros(m.shape, dtype=np.int32),
            count=0,
            areas=np.zeros(1, dtype=np.int64),
            slices=((slice(0, 0), slice(0, 0)),),
        )

    row, start, stop = _row_runs(m)
    n_runs = len(row)
    parent = _union_runs(row, start, stop, m.shape[1], bool(int(connectivity) == 8))

    def find(x: int) -> int:
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    roots = np.fromiter((find(i) for i in range(n_runs)), dtype=np.int64, count=n_runs)
    uniq, inverse = np.unique(roots, return_inverse=True)
    run_label = (inverse.reshape(-1) + 1).astype(np.int32)

    flat = np.flatnonzero(m.ravel())
    run_of_pixel = np.repeat(np.arange(n_runs, dtype=np.int64), stop - start)
    labels = np.zeros(m.size, dtype=np.int32)
    labels[flat] = run_label[run_of_pixel]
    labels = labels.reshape(m.shape)

    count = int(uniq.size)
    areas = np.bincount(labels.ravel(), minlength=count + 1).astype(np.int64)

    # 包围盒：在游程层面归约，避免对全图做 ufunc.at
    lab = run_label
    rmin = np.full(count + 1, m.shape[0], dtype=np.int64)
    rmax = np.full(count + 1, -1, dtype=np.int64)
    cmin = np.full(count + 1, m.shape[1], dtype=np.int64)
    cmax = np.full(count + 1, -1, dtype=np.int64)
    np.minimum.at(rmin, lab, row)
    np.maximum.at(rmax, lab, row)
    np.minimum.at(cmin, lab, start)
    np.maximum.at(cmax, lab, stop - 1)
    slices = tuple(
        (slice(0, 0), slice(0, 0)) if i == 0 else
        (slice(int(rmin[i]), int(rmax[i]) + 1), slice(int(cmin[i]), int(cmax[i]) + 1))
        for i in range(count + 1)
    )
    return Components(labels=labels, count=count, areas=areas, slices=slices)


def region_from_seeds(
    mask: np.ndarray,
    seeds: np.ndarray,
    connectivity: int = 4,
) -> np.ndarray:
    """从任意种子点出发可达的连通区域之和。

    等价于对每个种子各做一次 cv2.floodFill 后取并集，但只需一趟标记。
    cv2.floodFill 的默认连通性是 4，所以这里默认 connectivity=4。
    """
    m = _as_mask(mask)
    seeds = _as_mask(seeds) & m
    if not seeds.any():
        return np.zeros_like(m)
    comp = connected_components(m, connectivity=connectivity)
    ids = np.unique(comp.labels[seeds])
    ids = ids[ids > 0]
    return comp.mask_of(ids)
