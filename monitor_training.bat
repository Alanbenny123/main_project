@echo off
REM Monitor training progress
echo Checking training status...
echo.

python check_training_progress.py

echo.
echo ========================================
echo Latest training log (last 20 lines):
echo ========================================
powershell -Command "Get-Content training_behavior.log -Tail 20 -ErrorAction SilentlyContinue"

echo.
echo Press Ctrl+C to stop monitoring
timeout /t 30 > nul
goto :eof

