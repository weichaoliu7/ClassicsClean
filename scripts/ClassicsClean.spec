# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller 打包规格：单文件、无控制台窗口的 Windows GUI 可执行文件。

产物：dist/ClassicsClean.exe
  * 双击打开 -> 图形界面（支持拖拽 / 选择文件）
  * 带参数运行 -> 命令行批处理模式（ClassicsClean.exe in.pdf -o out.pdf）

体积策略（v1.2，99.2 MB -> 约 30 MB，行为逐像素不变）：
  1. 不打包 OpenCV：形态学 / 滤波 / 重采样算子全部由 classicsclean.morphology
     以纯 numpy 实现（与 cv2 逐像素等价，tests/unit/test_morphology.py 对照），
     cv2.pyd 单体 82 MB + ffmpeg 后端 12 MB 随之整体消失；
  2. Analysis 之后统一过滤 a.binaries / a.datas / a.pure：collect_all 与各钩子
     （PIL / pymupdf / tkinterdnd2 / numpy）整包收录的开发期文件在此剔除。
     前置的 collect_all 过滤挡不住钩子自己收录的文件（cv2 的 ffmpeg、PIL 的
     _avif 都曾因此漏掉），所以必须后置；
  3. UPX LZMA 压缩全部二进制（-55% ~ -85%）。

注意：入口脚本必须在包目录之外，否则包内相对导入会失败。
"""
import os
import sys

from PyInstaller.utils.hooks import collect_all

ROOT = os.path.abspath(os.path.join(SPECPATH, ".."))  # noqa: F821
SRC = os.path.join(ROOT, "src")
ENTRY = os.path.join(SPECPATH, "pyinstaller_entry.py")  # noqa: F821

datas, binaries, hiddenimports = [], [], []
for pkg in ("pymupdf", "pikepdf", "PIL", "tkinterdnd2"):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

# ---------- 体积削减：剔除明确用不到的二进制与数据 ----------
# 命中规则按源码路径子串匹配；每一项都在运行期验证过「import 链不触及」。
DROP_PATTERNS = (
    # OpenCV（v1.2 起算法不再依赖 cv2；这两项只是保险，正常什么都收不到）
    "/cv2/",                    # 82 MB：cv2.pyd 单体
    "opencv_videoio",           # 12 MB：OpenCV 的视频读写后端
    # Pillow：只走 JPEG2000 编解码，其余编解码 / 字体 / 色彩管理插件用不到
    "_avif",                    # 4.1 MB：AVIF 插件
    "_imagingcms",              # ICC 色彩管理扩展
    "_imagingft",               # FreeType 字体渲染（运行期 import 验证未触及）
    # PyMuPDF：C 开发库（.lib / .h），运行期用不到。
    # 注意不能剔 mupdf.py——pymupdf/__init__.py 顶层就要 from . import mupdf。
    "mupdf-devel",
    # numpy.random：本项目运行期 import 链验证不触及（np.random 只有测试夹具用）
    "numpy/random",
    # lxml：pikepdf 的静态依赖图顺带带入；PDF 装配 / 元数据 / 保存路径实测不需要
    "lxml",
    # tkinterdnd2：只保留 Windows x64 构建
    "tkdnd/win-arm64",
    "tkdnd/win-x86",
    # 注意：不能剔 numpy.libs/libscipy_openblas——numpy._core._multiarray_umath
    # 在 PE 导入表里静态依赖它，缺了连 import numpy 都失败（已实测）。
)


def _keep(item) -> bool:
    src = str(item[0]).replace("\\", "/")
    return not any(p.replace("\\", "/") in src for p in DROP_PATTERNS)


# Visual C++ 运行库：MuPDF / qpdf / numpy 均为 MSVC 构建，目标机可能缺少该 DLL。
# 该文件属于微软可再发行组件，随产品一起分发是被允许的。
for dll in ("msvcp140.dll", "msvcp140_1.dll", "msvcp140_2.dll", "vcruntime140.dll",
            "vcruntime140_1.dll"):
    for candidate in (
        os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32", dll),
        os.path.join(sys.base_prefix, dll),
    ):
        if os.path.isfile(candidate):
            binaries.append((candidate, "."))
            break

hiddenimports += [
    "classicsclean",
    "classicsclean.apps.cli",
    "classicsclean.apps.gui",
    "classicsclean.pipeline.document",
    "classicsclean.pipeline.page",
    "classicsclean.algorithms.masks",
    "classicsclean.algorithms.white_point",
    "classicsclean.algorithms.tone",
    "classicsclean.algorithms.repair",
]

a = Analysis(  # noqa: F821
    [ENTRY],
    pathex=[SRC],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["matplotlib", "scipy", "pandas", "IPython", "notebook", "PyQt5", "PySide6",
              "pytest", "tkinter.test",
              # 下面这些没有任何模块真正 import（运行时验证过），纯属被钩子顺带拖进来
              "PIL.ImageQt"],
    noarchive=False,
    optimize=2,
)

# ---------- 后置统一过滤：把钩子 / collect_all 整包带入、运行期零引用的文件剔除 ----------
before_binaries = len(a.binaries)
before_datas = len(a.datas)
before_pure = len(a.pure)
a.binaries = [b for b in a.binaries if _keep(b)]
a.datas = [d for d in a.datas if _keep(d)]
a.pure = [m for m in a.pure if _keep(m)]
print(f"[体积削减] 二进制 {before_binaries} -> {len(a.binaries)}，"
      f"数据 {before_datas} -> {len(a.datas)}，"
      f"纯模块 {before_pure} -> {len(a.pure)}")

pyz = PYZ(a.pure)  # noqa: F821

exe = EXE(  # noqa: F821
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="ClassicsClean",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,          # Windows 无 strip 工具，符号剥离由 UPX --strip-loadconf 承担
    upx=True,             # UPX LZMA 压缩全部二进制（需 .toolchain/upx/upx.exe 在 PATH）
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
