@echo off
chcp 65001 >nul
title GAME ONNX 便携环境一键配置
echo ========================================
echo   GAME ONNX 便携环境一键配置脚本
echo ========================================

set PYTHON_VERSION=3.10.11
set PYTHON_URL=https://registry.npmmirror.com/-/binary/python/%PYTHON_VERSION%/python-%PYTHON_VERSION%-embed-amd64.zip
set PYTHON_DIR=%~dp0python

if exist "%PYTHON_DIR%\python.exe" (
    echo [INFO] 便携版 Python 已存在，跳过下载步骤。
    goto :install_reqs
)

echo [INFO] 正在下载 Python %PYTHON_VERSION% 便携版 (使用国内镜像)...
powershell -Command "Invoke-WebRequest -Uri '%PYTHON_URL%' -OutFile 'python.zip'"
if not exist "python.zip" (
    echo [ERROR] 下载 Python 失败！请检查网络。
    pause
    exit /b
)

echo [INFO] 正在解压 Python...
powershell -Command "Expand-Archive -Path 'python.zip' -DestinationPath '%PYTHON_DIR%' -Force"
del python.zip

echo [INFO] 正在配置 pip...
:: 取消注释 python310._pth 中的 import site，以启用 pip 和第三方库支持
powershell -Command "(Get-Content '%PYTHON_DIR%\python310._pth') -replace '#import site', 'import site' | Set-Content '%PYTHON_DIR%\python310._pth'"

:: 下载并安装 pip
powershell -Command "Invoke-WebRequest -Uri 'https://bootstrap.pypa.io/get-pip.py' -OutFile '%PYTHON_DIR%\get-pip.py'"
"%PYTHON_DIR%\python.exe" "%PYTHON_DIR%\get-pip.py"

:install_reqs
echo [INFO] 正在安装 ONNX 推理及网页界面所需的依赖 (使用清华镜像源)...
"%PYTHON_DIR%\python.exe" -m pip install -r requirements_onnx.txt --index-url https://pypi.tuna.tsinghua.edu.cn/simple

echo.
echo ========================================
echo   环境配置完成！现在你可以双击“一键启动网页界面.bat”来运行程序了。
echo ========================================
pause
