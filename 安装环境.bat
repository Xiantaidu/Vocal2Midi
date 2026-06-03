@echo off
chcp 65001 >nul
setlocal ENABLEEXTENSIONS
title Vocal2Midi Portable Runtime Setup

set "ROOT=%~dp0"
cd /d "%ROOT%"

set "PYTHON_VERSION=3.12.10"
set "PYTHON_TAG=312"
set "PYTHON_DIR=%ROOT%python"
set "PYTHON_ARCHIVE=python-%PYTHON_VERSION%-embed-amd64.zip"
set "PYTHON_ZIP=%ROOT%%PYTHON_ARCHIVE%"
set "PYTHON_URL_MIRROR=https://registry.npmmirror.com/-/binary/python/%PYTHON_VERSION%/%PYTHON_ARCHIVE%"
set "PYTHON_URL_OFFICIAL=https://www.python.org/ftp/python/%PYTHON_VERSION%/%PYTHON_ARCHIVE%"
set "GET_PIP_URL=https://bootstrap.pypa.io/get-pip.py"
set "PIP_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple"
set "GET_PIP_FILE=%PYTHON_DIR%\get-pip.py"
set "PYTHON_PTH=%PYTHON_DIR%\python%PYTHON_TAG%._pth"
set "CORE_REQUIREMENTS=%ROOT%requirements_portable_core.txt"

echo ========================================
echo   Vocal2Midi portable runtime setup
echo ========================================
echo.

if not exist "%PYTHON_DIR%\python.exe" (
    echo [INFO] Downloading embeddable Python %PYTHON_VERSION%...
    powershell -NoProfile -ExecutionPolicy Bypass -Command ^
      "try { Invoke-WebRequest -UseBasicParsing -Uri '%PYTHON_URL_MIRROR%' -OutFile '%PYTHON_ZIP%' } catch { exit 1 }"
    if errorlevel 1 (
        echo [WARN] Mirror download failed, retrying from python.org...
        powershell -NoProfile -ExecutionPolicy Bypass -Command ^
          "Invoke-WebRequest -UseBasicParsing -Uri '%PYTHON_URL_OFFICIAL%' -OutFile '%PYTHON_ZIP%'"
        if errorlevel 1 (
            echo [ERROR] Failed to download embeddable Python.
            echo         You can also download it manually and extract it to:
            echo         %PYTHON_DIR%
            pause
            exit /b 1
        )
    )

    echo [INFO] Extracting embeddable Python...
    powershell -NoProfile -ExecutionPolicy Bypass -Command ^
      "Expand-Archive -Path '%PYTHON_ZIP%' -DestinationPath '%PYTHON_DIR%' -Force"
    if errorlevel 1 (
        echo [ERROR] Failed to extract Python archive.
        pause
        exit /b 1
    )
    del /f /q "%PYTHON_ZIP%" >nul 2>nul
) else (
    echo [INFO] Portable Python already exists, skipping download.
)

if exist "%PYTHON_PTH%" (
    echo [INFO] Enabling site-packages support...
    powershell -NoProfile -ExecutionPolicy Bypass -Command ^
      "(Get-Content '%PYTHON_PTH%') -replace '^#import site$', 'import site' | Set-Content '%PYTHON_PTH%'"
    powershell -NoProfile -ExecutionPolicy Bypass -Command ^
      "$lines = Get-Content '%PYTHON_PTH%';" ^
      "if ($lines -notcontains '..') {" ^
      "  $out = New-Object System.Collections.Generic.List[string];" ^
      "  foreach ($line in $lines) {" ^
      "    if ($line -eq 'import site') { $out.Add('..') }" ^
      "    $out.Add($line)" ^
      "  }" ^
      "  Set-Content '%PYTHON_PTH%' -Value $out" ^
      "}"
)

if not exist "%PYTHON_DIR%\Scripts\pip.exe" (
    echo [INFO] Installing pip...
    powershell -NoProfile -ExecutionPolicy Bypass -Command ^
      "Invoke-WebRequest -UseBasicParsing -Uri '%GET_PIP_URL%' -OutFile '%GET_PIP_FILE%'"
    if errorlevel 1 (
        echo [ERROR] Failed to download get-pip.py.
        pause
        exit /b 1
    )
    "%PYTHON_DIR%\python.exe" "%GET_PIP_FILE%"
    if errorlevel 1 (
        echo [ERROR] Failed to install pip.
        pause
        exit /b 1
    )
)

set "PIP_DISABLE_PIP_VERSION_CHECK=1"
set "PYTHONNOUSERSITE=1"

echo [INFO] Upgrading pip/setuptools/wheel...
"%PYTHON_DIR%\python.exe" -m pip install --upgrade pip setuptools wheel --index-url %PIP_INDEX%
if errorlevel 1 (
    echo [ERROR] Failed to upgrade bootstrap packages.
    pause
    exit /b 1
)

echo [INFO] Installing Vocal2Midi runtime dependencies...
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "Get-Content 'requirements.txt' | Where-Object { $_.Trim() -ne 'pyopenjtalk' } | Set-Content '%CORE_REQUIREMENTS%'"
if errorlevel 1 (
    echo [ERROR] Failed to prepare core requirements list.
    pause
    exit /b 1
)

"%PYTHON_DIR%\python.exe" -m pip install -r "%CORE_REQUIREMENTS%" --index-url %PIP_INDEX%
if errorlevel 1 (
    echo [ERROR] Failed to install required packages.
    del /f /q "%CORE_REQUIREMENTS%" >nul 2>nul
    pause
    exit /b 1
)
del /f /q "%CORE_REQUIREMENTS%" >nul 2>nul

echo [INFO] Installing optional Japanese G2P dependency: pyopenjtalk...
"%PYTHON_DIR%\python.exe" -m pip install pyopenjtalk --no-build-isolation --index-url %PIP_INDEX%
if errorlevel 1 (
    echo [WARN] pyopenjtalk installation failed.
    echo [WARN] The GUI can still run, but Japanese G2P features may be unavailable.
)

echo.
echo ========================================
echo   Setup complete.
echo   Now run: 启动GUI.bat
echo ========================================
pause
