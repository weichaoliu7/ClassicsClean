@echo off
chcp 65001 >nul
REM ============================================================
REM  ClassicsClean 一键质检
REM
REM      python scripts\run_tests.py --scope full --exe
REM
REM  依次完成：单元测试 -> 全书终检（真书）-> 成品 exe 验收
REM            -> 生成 reports\test-report.md（含全部结论）
REM
REM  用法：
REM      scripts\check_all.bat                          自动挑选解释器
REM      scripts\check_all.bat D:\path\to\python.exe    指定解释器
REM  只跑秒级单元测试：
REM      scripts\check_all.bat --quick
REM ============================================================
setlocal
set "PY=%~2"
if "%PY%"=="" (
    if exist "%~dp0..\.toolchain\python\python.exe" (
        set "PY=%~dp0..\.toolchain\python\python.exe"
    ) else (
        set "PY=python"
    )
)
set "SCOPE=full"
if "%1"=="--quick" set "SCOPE=unit"

pushd "%~dp0.."
echo 使用解释器：%PY%
"%PY%" scripts\run_tests.py --scope %SCOPE% --exe
if errorlevel 1 (
    echo [结论] 质检未通过，详见 reports\test-report.md
    popd
    exit /b 1
)
echo [结论] 全部通过，报告：reports\test-report.md
popd
endlocal
