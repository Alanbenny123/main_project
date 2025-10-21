# Student Behavior Analysis System 🎓

AI-powered classroom behavior detection using deep learning and computer vision.

## Features

- **📤 Upload Analysis**: Analyze images or videos to detect student behaviors
- **📹 Live Webcam Analysis**: Real-time behavior detection using your webcam
- **🎥 Full Classroom Analysis**: Multi-student tracking with face recognition
- **👤 Face Enrollment**: Register student faces for identification (Admin only)
- **📊 Student Dashboard**: Students can view their own reports (Login required)
- **📋 Audit Log**: Track all system activities and data changes (Admin only)
- **🔐 Authentication**: Separate login systems for students and admins
- **📊 Report Generation**: Automatic report creation with engagement scores
- **🗄️ Database Storage**: Store and retrieve analysis history
- **4 Behavior Classes**: 
  - ✋ Raising Hand
  - 📖 Reading
  - 😴 Sleeping
  - ✍️ Writing
- **Modern React UI**: Beautiful, vibrant interface with glass morphism
- **Flask Backend**: Robust API for model predictions and data management
- **Swin Transformer**: State-of-the-art vision transformer architecture

## Tech Stack

### Frontend
- React 18
- Vite
- Modern CSS with animations
- Light/Dark theme support

### Backend
- **Flask** - REST API server
- **TensorFlow/Keras** - Behavior classification (Swin Transformer)
- **YOLOv8n** - Person/object detection (Ultralytics)
- **InsightFace Buffalo_l** - Facial recognition (512D embeddings)
- **OpenCV** - Video processing
- **SQLite** - Database for reports and students
- **NumPy** - Numerical computations

## Setup Instructions

### 1. Backend Setup

```bash
# Install Python dependencies
pip install -r requirements.txt

# Run Flask API
python app.py
```

The API will run on `http://localhost:5000`

### 2. Frontend Setup

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

The React app will run on `http://localhost:3000`

### 3. Dataset Setup (Optional - for training)

1. Download dataset from [Kaggle](https://www.kaggle.com/datasets/phamluhuynhmai/classroom-student-behaviors)
2. Extract to `./data/Behaviors_Features/`
3. Run the notebook to train the model

## Project Structure

```
mma/
├── app.py                          # Flask backend API
├── requirements.txt                # Python dependencies
├── Student_Behaviors_NoteBook_1 (2).ipynb  # Training notebook
├── frontend/
│   ├── src/
│   │   ├── App.jsx                # Main React component
│   │   ├── components/
│   │   │   ├── UploadSection.jsx  # File upload component
│   │   │   ├── ResultsSection.jsx # Results display
│   │   │   └── StatsSection.jsx   # Dataset statistics
│   │   └── App.css                # Styles
│   └── package.json
└── data/
    └── Behaviors_Features/         # Dataset (not included)
```

## Usage

### Quick Start

**Windows:**
```bash
.\start.bat
```

**Linux/Mac:**
```bash
chmod +x start.sh
./start.sh
```

Or manually:
```bash
# Terminal 1 - Backend
python app.py

# Terminal 2 - Frontend
cd frontend
npm run dev
```

Open browser to `http://localhost:3000`

### Using the Application

#### 1. **Upload Analysis Tab**
- Drag & drop or click to upload an image/video
- Click "Analyze Behavior" button
- View predictions with confidence scores
- See detailed behavior distribution for videos

#### 2. **Live Analysis Tab**
- Enter Student ID (e.g., S001, S002, S003)
- Click "Start Camera" to begin webcam analysis
- System analyzes behavior in real-time (every 2 seconds)
- View live behavior counts and duration
- Click "Stop & Save Report" to save the session
- Add optional notes and confirm save

#### 3. **Full Classroom Analysis Tab**
- Upload classroom video or use live webcam
- System detects and identifies all students using face recognition
- Tracks individual student behaviors simultaneously
- Generate separate reports for each student
- View aggregated class statistics

#### 4. **Face Enrollment Tab (Admin Only)**
- **Login:** Default password is `admin123`
- **Webcam Capture:** Take multiple snapshots of student's face
- **Upload Existing:** Upload existing face photos for a student
- **Bulk Upload:** Upload ZIP file with folders per student (e.g., S001, S002)
- System requires admin authentication

#### 5. **Student Dashboard Tab (Login Required)**
- **Login:** Use Student ID (e.g., S001) and password (default: `password123`)
- View all your behavior analysis reports
- See engagement scores and behavior distribution
- Click "View Details" to see full report with frame-by-frame analysis
- Track progress over multiple sessions

#### 6. **Audit Log Tab (Admin Only)**
- Track all system activities (logins, enrollments, reports)
- See who created/modified data
- Monitor student and admin actions
- Filter by user type, action, or entity

### Initial Setup

**Create sample students:**
```bash
python database.py
```

This creates:
- **S001** - John Doe (password: `password123`)
- **S002** - Jane Smith (password: `password123`)
- **S003** - Bob Johnson (password: `password123`)

**Admin Credentials:**
- Password: `admin123`
- Change via Audit Log tab after first login

## API Endpoints

### Analysis
- `POST /api/predict` - Upload file for prediction
- `POST /api/analyze-frame` - Analyze single webcam frame
- `GET /api/stats` - Get dataset statistics
- `GET /api/model-info` - Get model information

### Student Management
- `GET /api/students` - List all students
- `GET /api/students/<student_id>` - Get student info
- `POST /api/students` - Create new student

### Authentication
- `POST /api/admin/login` - Admin login
- `POST /api/admin/logout` - Admin logout
- `GET /api/admin/status` - Check admin login status
- `POST /api/admin/change-password` - Change admin password
- `POST /api/student/login` - Student login
- `POST /api/student/logout` - Student logout
- `GET /api/student/status` - Check student login status

### Audit & Monitoring
- `GET /api/admin/audit-logs` - Get activity logs (Admin only)

### Reports
- `POST /api/reports/save` - Save analysis report
- `GET /api/reports/student/<student_id>` - Get student's reports
- `GET /api/reports/<report_id>` - Get detailed report

## Model & Algorithm Information

### Behavior Classification
- **Architecture**: Swin Transformer (Vision Transformer)
- **Input Size**: 224×224×3 RGB images
- **Classes**: 4 (Raising Hand, Reading, Sleeping, Writing)
- **Training Samples**: 460
- **Framework**: TensorFlow 2.x with Keras

### Object Detection
- **Model**: YOLOv8n (Ultralytics)
- **Purpose**: Person detection in classroom videos
- **Speed**: Real-time capable (100+ FPS on GPU)
- **Accuracy**: High precision for person bounding boxes
- **Input**: Variable size images (auto-scaled to 640×640)

### Face Recognition
- **Model**: InsightFace Buffalo_l
- **Embedding Size**: 512 dimensions
- **Method**: Cosine similarity matching
- **Similarity Threshold**: 0.4 (adjustable)
- **Accuracy**: 99.8% on LFW benchmark
- **Speed**: ~50ms per face on CPU, ~10ms on GPU
- **Validation Samples**: 116
- **Test Samples**: 144

## Development

### Build for Production

```bash
cd frontend
npm run build
```

The built files will be in `frontend/dist/`

### Train Model

Open and run `Student_Behaviors_NoteBook_1 (2).ipynb` in Jupyter:

```bash
jupyter notebook
```

## Adding the Model

The app works in **demo mode** without a trained model, but for real predictions:

### Quick Guide:
1. Download dataset from [Kaggle](https://www.kaggle.com/datasets/phamluhuynhmai/classroom-student-behaviors)
2. Extract to `./data/Behaviors_Features/`
3. Open and run `Student_Behaviors_NoteBook_1 (2).ipynb` in Jupyter
4. Save trained model to `./saved_model/student_behavior_model.h5`
5. Restart Flask backend

**See [MODEL_SETUP.md](MODEL_SETUP.md) for detailed instructions.**

## Demo Mode

Without a trained model, the app runs in demo mode with realistic mock predictions for testing the UI.

## License

This project is for educational purposes.
# main_project
# main_project
