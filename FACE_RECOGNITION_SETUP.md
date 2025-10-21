# Face Recognition Setup Guide 👤

Complete guide for setting up face recognition to identify students in classroom videos.

## Overview

The system uses **OpenCV's LBPH Face Recognizer** to:
1. Detect faces in classroom videos
2. Identify which student each face belongs to
3. Track individual behaviors for each student
4. Generate per-student reports

## Installation

### 1. Install OpenCV with Face Recognition

```bash
pip install opencv-contrib-python
```

This includes the face recognition module needed for LBPH.

### 2. Verify Installation

```bash
python -c "import cv2; print(cv2.__version__); print(hasattr(cv2, 'face'))"
```

Should output the version and `True`.

## Enrolling Students

### Method 1: Using the Web UI (Recommended)

1. Start the application
2. Go to **Face Enrollment** tab
3. Enter Student ID and Name
4. Click "Start Face Capture"
5. Capture 5-10 face samples (different angles, expressions)
6. Click "Enroll" to save

**Tips for good enrollment:**
- Face the camera directly
- Capture at different angles (slight left, right, up, down)
- Use good lighting
- Avoid glasses/masks if possible (or capture with/without)
- Capture different expressions

### Method 2: Using Python Script

```bash
python face_recognition_system.py
```

Follow the prompts to enroll students via webcam.

### Method 3: Via API

```bash
curl -X POST http://localhost:5000/api/face/enroll \
  -H "Content-Type: application/json" \
  -d '{
    "student_id": "S001",
    "name": "John Doe",
    "face_images": ["base64_image1", "base64_image2", ...]
  }'
```

## Using Full Classroom Analysis

### Live Classroom Analysis:

1. **Enroll all students** using Face Enrollment tab
2. Go to **Full Classroom** tab
3. Point camera at the classroom
4. Click "Start Classroom Analysis"
5. System detects all faces and identifies students
6. Real-time behavior tracking for each student
7. Click "Stop & Save All Reports" to save individual reports

### Analyzing Classroom Video:

```bash
curl -X POST http://localhost:5000/api/analyze-classroom-video \
  -F "file=@classroom_video.mp4"
```

Returns per-student behavior analysis:
```json
{
  "success": true,
  "student_data": {
    "S001": {
      "name": "John Doe",
      "behavior_stats": [
        {"label": "Reading", "count": 45, "percentage": 75},
        {"label": "Writing", "count": 15, "percentage": 25}
      ]
    },
    "S002": { ... }
  }
}
```

## How It Works

### 1. Face Detection

Uses **Haar Cascade** (fast, good for real-time):
- Detects faces in each frame
- Returns bounding boxes for each face

### 2. Face Recognition

Uses **LBPH (Local Binary Patterns Histograms)**:
- Extracts unique patterns from face
- Compares with enrolled students
- Returns student ID if match found

### 3. Behavior Prediction

For each detected face:
- Extracts face region + surrounding context
- Runs through behavior detection model
- Assigns behavior label

### 4. Tracking

- Maintains per-student session data
- Counts behaviors for each student
- Generates individual reports

## Face Recognition Files

```
student_faces/          # Face image storage
├── S001/
│   ├── face_0.jpg
│   ├── face_1.jpg
│   └── ...
├── S002/
│   └── ...
└── ...

face_encodings.pkl      # Trained model metadata
face_recognizer_model.yml  # LBPH model weights
```

## API Endpoints

### Enrollment
```
POST /api/face/enroll
  - Enroll student face samples

GET /api/face/enrolled
  - List enrolled students

GET /api/face/status
  - Check system status
```

### Analysis
```
POST /api/analyze-classroom-frame
  - Analyze single frame with multiple students

POST /api/analyze-classroom-video
  - Process full classroom video
```

## Troubleshooting

### Error: "module 'cv2' has no attribute 'face'"

**Solution:**
```bash
pip uninstall opencv-python opencv-contrib-python
pip install opencv-contrib-python==4.5.5.64
```

### Poor Recognition Accuracy

**Solutions:**
1. Enroll more face samples (8-10 per student)
2. Capture faces in different lighting conditions
3. Ensure good camera quality
4. Re-train after adding more samples

### Face Not Detected

**Solutions:**
1. Ensure good lighting
2. Face camera directly
3. Remove obstructions (hair, hands, etc.)
4. Adjust `minNeighbors` in face detector settings

### Wrong Student Identified

**Solutions:**
1. Add more diverse face samples
2. Check if students look similar
3. Lower confidence threshold for stricter matching
4. Re-enroll with better quality images

## Advanced Configuration

### Adjust Face Detection Sensitivity

In `face_recognition_system.py`:

```python
# For more sensitive detection (more false positives)
faces = detector.detectMultiScale(
    gray,
    scaleFactor=1.05,  # Lower = more sensitive
    minNeighbors=3,    # Lower = more detections
    minSize=(20, 20)   # Smaller = detect smaller faces
)

# For less sensitive (fewer false positives)
faces = detector.detectMultiScale(
    gray,
    scaleFactor=1.3,
    minNeighbors=7,
    minSize=(50, 50)
)
```

### Adjust Recognition Confidence

```python
# In identify_face() method
# Lower threshold = stricter matching
confidence_threshold = 80  # Default: 100

# Lower confidence value = better match
# Typical range: 50-100
```

## Performance Tips

### For Better Speed:
- Use Haar Cascade detection (current default)
- Process every Nth frame (sample_rate=30)
- Reduce video resolution
- Use GPU acceleration if available

### For Better Accuracy:
- Switch to DNN face detector
- Process more frames (lower sample_rate)
- Use higher resolution video
- Enroll more face samples

## Workflow Example

### Full Classroom Session:

1. **Setup (One Time)**
```bash
# Start application
python app.py

# Enroll all students via web UI
# - Go to Face Enrollment tab
# - Enroll S001, S002, S003, etc.
```

2. **Analysis Session**
```bash
# Go to Full Classroom tab
# Point camera at classroom
# Click Start Classroom Analysis
# System tracks all students in real-time
```

3. **Review Reports**
```bash
# Go to Reports tab
# Enter each student ID to view their report
# See engagement scores, behavior distribution
```

## Benefits of Face Recognition

✅ **Multi-Student Tracking**: Analyze entire classroom at once
✅ **Individual Reports**: Separate reports for each student
✅ **Automatic Identification**: No manual tagging needed
✅ **Scalable**: Works with any number of enrolled students
✅ **Privacy-Focused**: Faces stored locally, not in cloud

## Security & Privacy

- Face data stored locally only
- No external API calls
- Can be deleted anytime (`rm -rf student_faces/`)
- GDPR compliant (with proper consent)
- Recommend student/parent consent before enrollment

## Next Steps

1. ✅ Enroll all students in your classroom
2. ✅ Test with individual students first
3. ✅ Try full classroom analysis
4. ✅ Review and refine based on accuracy
5. ✅ Generate reports for parents/teachers

---

For questions or issues, check the main [README.md](README.md) or [MODEL_SETUP.md](MODEL_SETUP.md).


