@echo off
title Asta AI Launcher

where python >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python tidak ditemukan. Install Python 3.10+ dulu.
    pause
    exit /b 1
)

where node >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Node.js tidak ditemukan. Install Node.js 18+ dulu.
    pause
    exit /b 1
)

python "%~dp0setup_and_run.py"
if errorlevel 1 (
    echo [ERROR] Setup gagal.
    pause
    exit /b 1
)