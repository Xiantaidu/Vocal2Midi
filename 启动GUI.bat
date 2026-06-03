@echo off
chcp 65001 >nul
setlocal
title Vocal2Midi GUI

set "ROOT=%~dp0"
set "PYTHON_DIR=%ROOT%python"

if not exist "%PYTHON_DIR%\python.exe" (
    echo [ERROR] Portable Python was not found.
    echo         Please run 安装环境.bat first.
    pause
    exit /b 1
)

cd /d "%ROOT%"

set "V2M_PORTABLE_ROOT=%ROOT:~0,-1%"
set "PYTHONHOME=%PYTHON_DIR%"
set "PYTHONPATH=%ROOT%"
set "PYTHONNOUSERSITE=1"
set "PATH=%PYTHON_DIR%;%PYTHON_DIR%\Scripts;%PYTHON_DIR%\DLLs;%PYTHON_DIR%\Lib\site-packages\PyQt5\Qt5\bin;%ROOT%inference\qwen3asr_dml\bin;%PATH%"
set "QT_PLUGIN_PATH=%PYTHON_DIR%\Lib\site-packages\PyQt5\Qt5\plugins"
set "QT_QPA_PLATFORM_PLUGIN_PATH=%PYTHON_DIR%\Lib\site-packages\PyQt5\Qt5\plugins\platforms"

"%PYTHON_DIR%\python.exe" app_fluent.py
set "EXIT_CODE=%ERRORLEVEL%"

if not "%EXIT_CODE%"=="0" (
    echo.
    echo [ERROR] GUI exited with code %EXIT_CODE%.
    pause
)

exit /b %EXIT_CODE%
