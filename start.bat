@echo off
chcp 65001 >nul
title GAME 生成式自适应 MIDI 提取器 (ONNX)
set PYTHON_DIR=%~dp0python

if not exist "%PYTHON_DIR%\python.exe" (
    echo [ERROR] 找不到 Python 环境！
    echo 请确认你已经完整解压了整个整合包，且 python 文件夹没有被误删。
    pause
    exit /b
)

echo 正在启动 GAME Gradio 界面...
"%PYTHON_DIR%\python.exe" app.py

pause
