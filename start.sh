#!/bin/bash

echo "===================================="
echo "Student Behavior Analysis System"
echo "===================================="
echo ""

echo "Starting Flask Backend..."
python app.py &
BACKEND_PID=$!

sleep 3

echo "Starting React Frontend..."
cd frontend
npm run dev &
FRONTEND_PID=$!

echo ""
echo "===================================="
echo "Both servers are running!"
echo "Backend: http://localhost:5000"
echo "Frontend: http://localhost:3000"
echo "===================================="
echo ""
echo "Press Ctrl+C to stop both servers"

# Wait for user interrupt
trap "kill $BACKEND_PID $FRONTEND_PID; exit" INT
wait


