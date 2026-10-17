@echo off
chcp 65001 >nul
REM ============================================================
REM  ClassicsClean 一键打包（单文件 exe）
REM
REM  用法：
REM      scripts\build_exe.bat                          自动挑选解释器
REM      scripts\build_exe.bat D:\path\to\python.exe    指定解释器
REM
REM  前置条件（该 Python 必须带 tkinter，否则打不出图形界面）：
REM      python -m pip install -r requirements-dev.txt
REM
REM  体积优化（scripts\ClassicsClean.spec 内有完整说明）：
REM      - 算法不依赖 OpenCV（纯 numpy morphology），cv2 整体不入包
REM      - UPX LZMA 压缩二进制：需要 .toolchain\upx\upx.exe（缺失则自动跳过压缩）
REM
REM  产物：dist\ClassicsClean.exe（Win10/Win11 x64 单文件，免安装）
REM ============================================================
setlocal
set "PY=%~1"
if "%PY%"=="" (
    if exist "%~dp0..\.toolchain\python\python.exe" (
        set "PY=%~dp0..\.toolchain\python\python.exe"
    ) else (
        set "PY=python"
    )
)
echo 使用解释器：%PY%

REM UPX：放进 PATH，PyInstaller 会自动探测（找不到也不影响打包，只是不压缩）
if exist "%~dp0..\.toolchain\upx\upx.exe" (
    set "PATH=%~dp0..\.toolchain\upx;%PATH%"
    echo UPX：%~dp0..\.toolchain\upx\upx.exe
) else (
    echo [提示] 未找到 .toolchain\upx\upx.exe，跳过 UPX 压缩（exe 会大 10MB 左右）
)

pushd "%~dp0.."
"%PY%" -c "import PyInstaller" >nul 2>&1
if errorlevel 1 (
    echo [错误] 当前解释器缺少 PyInstaller。
    echo        请先执行： "%PY%" -m pip install -r requirements-dev.txt
    popd
    exit /b 1
)

echo [1/3] 清理旧工作目录 + 打包中（约 3-8 分钟，请耐心等待）
REM 用 cmd 自带的 rmdir 清工作目录（避免 PyInstaller --clean 触发第三方 Python 的
REM 文件删除保护而中止）；PyInstaller 自身的缓存目录保留，可加速增量重打包。
if exist ".cache\pyinstaller" rmdir /s /q ".cache\pyinstaller"
"%PY%" -m PyInstaller --noconfirm ^
    --workpath .cache\pyinstaller --distpath dist ^
    scripts\ClassicsClean.spec
if errorlevel 1 (
    echo [错误] 打包失败
    popd
    exit /b 1
)

echo [2/3] 产物
dir dist\ClassicsClean.exe

echo [3/3] 验收 exe（自检 + 真实转码 + 输出格式检查）
"%PY%" scripts\verify_exe.py
if errorlevel 1 (
    echo [错误] exe 验收未通过，详见 reports\exe-verify.json
    popd
    exit /b 1
)
echo 完成：%~dp0..\dist\ClassicsClean.exe
popd
endlocal
