# Quick Start Guide 🚀

Get the Student Behavior Analysis System running in 5 minutes!

## Prerequisites

- Python 3.7+ installed
- Node.js 16+ installed
- A webcam (for live analysis)

## Installation

### 1. Install Python Dependencies

```bash
pip install flask flask-cors opencv-python numpy
```

### 2. Install Frontend Dependencies

```bash
cd frontend
npm install
cd ..
```

### 3. Initialize Database

```bash
python database.py
```

This creates sample students: S001, S002, S003

## Run the Application

### Easy Way (Windows):

```bash
.\start.bat
```

### Easy Way (Linux/Mac):

```bash
chmod +x start.sh
./start.sh
```

### Manual Way:

**Terminal 1 - Backend:**
```bash
python app.py
```

**Terminal 2 - Frontend:**
```bash
cd frontend
npm run dev
```

## Access the App

Open your browser to: **http://localhost:3000**

You should see:
- 🟢 Green status = Model loaded (real predictions)
- 🔴 Red status = Demo mode (mock predictions)

## Try It Out!

### Test Upload Analysis:
1. Go to **Upload Analysis** tab
2. Upload any classroom image/video
3. Click "Analyze Behavior"
4. See the results!

### Test Live Analysis:
1. Go to **Live Analysis** tab
2. Enter Student ID: `S001`
3. Click "Start Camera" (allow camera access)
4. Watch real-time behavior detection
5. Click "Stop & Save Report"
6. Add notes and save

### View Reports:
1. Go to **Student Dashboard** tab
2. Enter Student ID: `S001`
3. Click "View Reports"
4. See all saved reports and engagement scores

## Common Issues

### "API Offline" error:
- Make sure Flask backend is running: `python app.py`

### "Student not found" error:
- Run `python database.py` to create sample students
- Or create a new student via the API

### Camera not working:
- Allow camera permissions in your browser
- Check if another app is using the camera

### Port already in use:
- Backend: Change port in `app.py`: `app.run(port=5001)`
- Frontend: Change in `frontend/vite.config.js`

## Next Steps

- ✅ Add your own students
- ✅ Try live analysis with different behaviors
- ✅ View generated reports
- ✅ Train your own model (see [MODEL_SETUP.md](MODEL_SETUP.md))

## Need Help?

- Check [README.md](README.md) for detailed documentation
- See [MODEL_SETUP.md](MODEL_SETUP.md) for model training
- Ensure both backend and frontend are running

Enjoy! 🎓


