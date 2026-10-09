@echo off
title Fake News Detection Web App
echo ========================================================
echo  Starting Fake News Detection App on Localhost...
echo ========================================================
echo.
cd /d "%~dp0"
timeout /t 2 /nobreak >nul
start http://localhost:5000
python app.py
pause
