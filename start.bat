@echo off
echo ====================================
echo Student Behavior Analysis System
echo ====================================
echo.

echo Starting Flask Backend...
start cmd /k "python app.py"

timeout /t 3

echo Starting React Frontend...
cd frontend
start cmd /k "npm run dev"

echo.
echo ====================================
echo Both servers are starting!
echo Backend: http://localhost:5000
echo Frontend: http://localhost:3000
echo ====================================


