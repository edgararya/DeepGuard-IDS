@echo off
title DeepGuard Backend [Docker Container]
echo ========================================================
echo   DeepGuard Backend - Starting Docker Container
echo   FastAPI REST API on http://127.0.0.1:8000
echo ========================================================
echo.

cd /d "%~dp0"
docker compose up --build

pause
