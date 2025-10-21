# Backend Architecture Documentation

Complete guide to the Flask-based backend of the Student Behavior Analysis System.

---

## 🏗️ Architecture Overview

**Framework**: Flask 2.3+  
**Language**: Python 3.8+  
**Database**: SQLite3  
**ML Framework**: TensorFlow 2.x  
**Computer Vision**: OpenCV, YOLOv8n, InsightFace  
**API Style**: RESTful  

---

## 📁 Project Structure

```
backend/
├── app.py                          # Main Flask application
├── database.py                     # SQLite database operations
├── face_recognition_system.py     # YOLO + InsightFace integration
├── admin_auth.py                   # Admin authentication
├── requirements.txt                # Python dependencies
├── behavior_analysis.db            # SQLite database (auto-created)
├── saved_model/
│   └── student_behavior_model.h5   # Swin Transformer model
├── student_faces/                  # Enrolled face images
│   ├── S001/
│   │   ├── face_0.jpg
│   │   └── face_1.jpg
│   └── S002/
├── uploads/                        # Temporary file uploads
├── face_encodings_buffalo.pkl      # InsightFace embeddings
└── .admin_password                 # Hashed admin password
```

---

## 🧩 Module Breakdown

### 1. **app.py** - Main Flask Application

**Purpose**: REST API server, request handling, model orchestration

#### Initialization
```python
from flask import Flask, request, jsonify, session
from flask_cors import CORS

app = Flask(__name__)
CORS(app, supports_credentials=True)
app.secret_key = os.urandom(24)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB
```

#### Global Variables
```python
model = None                    # Swin Transformer
face_recognizer = None          # InsightFace Buffalo_l
classroom_analyzer = None       # YOLO + InsightFace
LABELS = ['Raising Hand', 'Reading', 'Sleeping', 'Writing']
```

#### Model Loading
```python
def load_model():
    global model, face_recognizer, classroom_analyzer
    
    # Load Swin Transformer
    if os.path.exists('./saved_model/student_behavior_model.h5'):
        model = tf.keras.models.load_model(model_path)
    
    # Initialize face recognition
    face_recognizer = FaceRecognizer()
    classroom_analyzer = ClassroomAnalyzer(predict_from_frame)
```

#### Image Preprocessing
```python
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0  # Normalize to [0, 1]
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img
```

#### Behavior Prediction
```python
def predict_behavior(image_path):
    img = preprocess_image(image_path)
    predictions = model.predict(img)[0]
    
    pred_list = [
        {'label': LABELS[i], 'confidence': float(predictions[i])}
        for i in range(len(LABELS))
    ]
    pred_list.sort(key=lambda x: x['confidence'], reverse=True)
    
    return {
        'predictions': pred_list,
        'top_prediction': pred_list[0]['label'],
        'confidence': pred_list[0]['confidence']
    }
```

#### Video Processing
```python
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    predictions = []
    frame_count = 0
    sample_rate = 10  # Analyze every 10th frame
    
    while cap.isOpened() and frame_count < 100:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % sample_rate == 0:
            result = predict_behavior(temp_frame_path)
            predictions.append(result)
        
        frame_count += 1
    
    cap.release()
    
    # Aggregate statistics
    behavior_stats = calculate_statistics(predictions)
    return behavior_stats
```

---

### 2. **database.py** - Data Persistence

**Purpose**: SQLite operations for students, reports, and audit logs

#### Database Schema

##### Students Table
```sql
CREATE TABLE students (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id TEXT UNIQUE NOT NULL,
    name TEXT NOT NULL,
    email TEXT,
    password_hash TEXT,  -- SHA256 hash
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
```

##### Reports Table
```sql
CREATE TABLE reports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id TEXT NOT NULL,
    session_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    duration INTEGER,
    total_frames INTEGER,
    raising_hand_count INTEGER DEFAULT 0,
    reading_count INTEGER DEFAULT 0,
    sleeping_count INTEGER DEFAULT 0,
    writing_count INTEGER DEFAULT 0,
    raising_hand_percent REAL DEFAULT 0,
    reading_percent REAL DEFAULT 0,
    sleeping_percent REAL DEFAULT 0,
    writing_percent REAL DEFAULT 0,
    engagement_score REAL,
    notes TEXT,
    FOREIGN KEY (student_id) REFERENCES students (student_id)
)
```

##### Frame Analysis Table
```sql
CREATE TABLE frame_analysis (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    report_id INTEGER NOT NULL,
    frame_number INTEGER,
    timestamp REAL,
    behavior TEXT,
    confidence REAL,
    FOREIGN KEY (report_id) REFERENCES reports (id)
)
```

##### Audit Log Table
```sql
CREATE TABLE audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    user_type TEXT NOT NULL,  -- 'admin' or 'student'
    action TEXT NOT NULL,
    entity_type TEXT,
    entity_id TEXT,
    details TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
```

#### Key Functions

##### Add Student
```python
def add_student(student_id, name, email=None, password=None):
    import hashlib
    conn = sqlite3.connect('behavior_analysis.db')
    cursor = conn.cursor()
    
    password_hash = None
    if password:
        password_hash = hashlib.sha256(password.encode()).hexdigest()
    
    cursor.execute(
        'INSERT INTO students (student_id, name, email, password_hash) VALUES (?, ?, ?, ?)',
        (student_id, name, email, password_hash)
    )
    conn.commit()
    conn.close()
```

##### Save Report
```python
def save_report(student_id, duration, behavior_stats, frame_data, notes=None):
    conn = sqlite3.connect('behavior_analysis.db')
    cursor = conn.cursor()
    
    # Calculate engagement score
    engagement_score = (
        behavior_percents.get('Raising Hand', 0) * 1.0 +
        behavior_percents.get('Writing', 0) * 0.9 +
        behavior_percents.get('Reading', 0) * 0.8 +
        behavior_percents.get('Sleeping', 0) * 0.1
    ) / 100
    
    cursor.execute('''
        INSERT INTO reports (...) VALUES (...)
    ''', (...))
    
    report_id = cursor.lastrowid
    
    # Save frame-by-frame data
    for frame in frame_data:
        cursor.execute('''
            INSERT INTO frame_analysis (...) VALUES (...)
        ''', (...))
    
    conn.commit()
    conn.close()
    
    return report_id
```

##### Verify Password
```python
def verify_student_password(student_id, password):
    import hashlib
    conn = sqlite3.connect('behavior_analysis.db')
    cursor = conn.cursor()
    
    cursor.execute('SELECT password_hash FROM students WHERE student_id = ?', (student_id,))
    result = cursor.fetchone()
    conn.close()
    
    if not result or not result[0]:
        return False
    
    password_hash = hashlib.sha256(password.encode()).hexdigest()
    return password_hash == result[0]
```

---

### 3. **face_recognition_system.py** - Computer Vision

**Purpose**: YOLO person detection + InsightFace face recognition

#### YOLOv8n Detector
```python
class YOLODetector:
    def __init__(self, model_size='n', conf_threshold=0.5):
        self.model = YOLO(f'yolov8{model_size}.pt')
    
    def detect_persons(self, frame):
        results = self.model(frame, conf=0.5, classes=[0])  # class 0 = person
        
        persons = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0].cpu().numpy())
            persons.append((int(x1), int(y1), int(x2), int(y2), conf))
        
        return persons
```

#### InsightFace Recognizer
```python
class BuffaloFaceRecognizer:
    def __init__(self):
        self.app = FaceAnalysis(name='buffalo_l')
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        self.known_faces = {}
    
    def get_face_embedding(self, face_img):
        faces = self.app.get(face_img)
        if len(faces) > 0:
            return faces[0].embedding  # 512D vector
        return None
    
    def enroll_student(self, student_id, name, face_images):
        embeddings = []
        for img in face_images:
            embedding = self.get_face_embedding(img)
            if embedding is not None:
                embeddings.append(embedding)
        
        # Average embeddings
        avg_embedding = np.mean(embeddings, axis=0)
        
        self.known_faces[student_id] = {
            'name': name,
            'embedding': avg_embedding
        }
    
    def identify_face(self, face_img, similarity_threshold=0.4):
        query_embedding = self.get_face_embedding(face_img)
        
        best_match = None
        best_similarity = 0
        
        for student_id, data in self.known_faces.items():
            # Cosine similarity
            similarity = np.dot(query_embedding, data['embedding']) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(data['embedding'])
            )
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = student_id
        
        if best_similarity >= similarity_threshold:
            return best_match, float(best_similarity)
        
        return None, float(best_similarity)
```

#### Classroom Analyzer
```python
class ClassroomAnalyzer:
    def __init__(self, behavior_predictor_func):
        self.yolo_detector = YOLODetector(model_size='n')
        self.face_recognizer = BuffaloFaceRecognizer()
        self.behavior_predictor = behavior_predictor_func
    
    def analyze_frame(self, frame):
        results = []
        
        # Detect persons
        persons = self.yolo_detector.detect_persons(frame)
        
        for (x1, y1, x2, y2, conf) in persons:
            person_img = frame[y1:y2, x1:x2]
            
            # Identify student
            student_id, similarity = self.face_recognizer.identify_face(person_img)
            
            # Predict behavior
            behavior, behavior_conf = self.behavior_predictor(person_img)
            
            results.append({
                'student_id': student_id or 'Unknown',
                'behavior': behavior,
                'confidence': behavior_conf
            })
        
        return results
```

---

### 4. **admin_auth.py** - Authentication

**Purpose**: Session-based admin authentication

#### Password Hashing
```python
import hashlib

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(password, password_hash):
    return hash_password(password) == password_hash
```

#### Login/Logout
```python
def login_admin(username, password):
    if username == ADMIN_USERNAME and verify_password(password, ADMIN_PASSWORD_HASH):
        session['admin_logged_in'] = True
        session['admin_token'] = secrets.token_hex(16)
        return True, "Login successful"
    return False, "Invalid credentials"

def logout_admin():
    session.pop('admin_logged_in', None)
    session.pop('admin_token', None)
```

#### Protected Routes
```python
def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not is_admin_logged_in():
            return jsonify({'error': 'Admin authentication required'}), 401
        return f(*args, **kwargs)
    return decorated_function

@app.route('/api/admin/audit-logs')
@admin_required
def get_audit_logs():
    # Only accessible if admin logged in
    ...
```

---

## 🌐 API Endpoints

### Analysis Endpoints

#### POST /api/predict
Upload image/video for prediction
```python
@app.route('/api/predict', methods=['POST'])
def predict():
    file = request.files['file']
    
    if file_ext in {'mp4', 'avi', 'mov'}:
        result = process_video(filepath)
    else:
        result = predict_behavior(filepath)
    
    return jsonify({'success': True, 'result': result})
```

#### POST /api/analyze-frame
Analyze single webcam frame
```python
@app.route('/api/analyze-frame', methods=['POST'])
def analyze_frame():
    data = request.json
    image_data = data.get('frame')  # Base64 encoded
    
    # Decode
    img_bytes = base64.b64decode(image_data.split('base64,')[1])
    nparr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Predict
    behavior, confidence = predict_from_frame(frame)
    
    return jsonify({'success': True, 'behavior': behavior, 'confidence': confidence})
```

### Student Management

#### GET /api/students
List all students
```python
@app.route('/api/students', methods=['GET'])
def list_students():
    students = get_all_students()
    return jsonify({'students': students})
```

#### POST /api/students
Create new student
```python
@app.route('/api/students', methods=['POST'])
def create_student():
    data = request.json
    student_id = data.get('student_id')
    name = data.get('name')
    password = data.get('password')
    
    success = add_student(student_id, name, None, password)
    return jsonify({'message': 'Student created successfully'})
```

### Authentication

#### POST /api/admin/login
Admin login
```python
@app.route('/api/admin/login', methods=['POST'])
def admin_login():
    data = request.json
    username = data.get('username')
    password = data.get('password')
    
    success, message = login_admin(username, password)
    
    if success:
        return jsonify({'success': True})
    return jsonify({'error': message}), 401
```

#### POST /api/student/login
Student login
```python
@app.route('/api/student/login', methods=['POST'])
def student_login():
    data = request.json
    student_id = data.get('student_id')
    password = data.get('password')
    
    if verify_student_password(student_id, password):
        session['student_logged_in'] = True
        session['student_id'] = student_id
        student = get_student(student_id)
        return jsonify({'success': True, 'student': student})
    
    return jsonify({'error': 'Invalid credentials'}), 401
```

### Face Recognition

#### POST /api/face/enroll
Enroll student faces (Admin only)
```python
@app.route('/api/face/enroll', methods=['POST'])
@admin_required
def enroll_student_face():
    data = request.json
    student_id = data.get('student_id')
    name = data.get('name')
    face_images_b64 = data.get('face_images')
    
    # Decode images
    face_images = []
    for img_b64 in face_images_b64:
        img_bytes = base64.b64decode(img_b64.split('base64,')[1])
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        face_images.append(img)
    
    # Enroll
    face_recognizer.enroll_student(student_id, name, face_images)
    face_recognizer.train()
    
    return jsonify({'success': True})
```

#### POST /api/admin/bulk-upload-faces
Bulk face upload via ZIP (Admin only)
```python
@app.route('/api/admin/bulk-upload-faces', methods=['POST'])
@admin_required
def bulk_upload_faces():
    file = request.files['file']
    
    # Extract ZIP
    with zipfile.ZipFile(upload_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    
    # Process folders: S001_JohnDoe/face1.jpg, face2.jpg...
    for folder_name in os.listdir(extract_path):
        student_id = folder_name.split('_')[0]
        name = '_'.join(folder_name.split('_')[1:])
        
        # Load images
        face_images = []
        for img_file in os.listdir(folder_path):
            img = cv2.imread(img_path)
            face_images.append(img)
        
        # Enroll
        face_recognizer.enroll_student(student_id, name, face_images)
    
    face_recognizer.train()
    return jsonify({'success': True, 'enrolled_count': enrolled_count})
```

### Classroom Analysis

#### POST /api/analyze-classroom-frame
Analyze classroom frame with multiple students
```python
@app.route('/api/analyze-classroom-frame', methods=['POST'])
def analyze_classroom_frame():
    data = request.json
    image_data = data.get('frame')
    
    # Decode
    frame = decode_base64_image(image_data)
    
    # Analyze (YOLO + InsightFace)
    detections = classroom_analyzer.analyze_frame(frame)
    
    return jsonify({
        'success': True,
        'detections': detections,
        'count': len(detections)
    })
```

### Reports

#### POST /api/reports/save
Save analysis report
```python
@app.route('/api/reports/save', methods=['POST'])
def save_analysis_report():
    data = request.json
    student_id = data.get('student_id')
    duration = data.get('duration')
    behavior_stats = data.get('behavior_stats')
    frame_data = data.get('frame_data')
    
    report_id = save_report(student_id, duration, behavior_stats, frame_data)
    
    return jsonify({
        'success': True,
        'report_id': report_id
    })
```

#### GET /api/reports/student/<student_id>
Get student's reports
```python
@app.route('/api/reports/student/<student_id>', methods=['GET'])
def get_student_report_list(student_id):
    reports = get_student_reports(student_id)
    return jsonify({'reports': reports})
```

---

## 🔐 Security Features

### 1. Password Hashing
```python
# SHA256 (should upgrade to bcrypt in production)
password_hash = hashlib.sha256(password.encode()).hexdigest()
```

### 2. Session Management
```python
app.secret_key = os.urandom(24)  # Random secret key
session['admin_logged_in'] = True
```

### 3. CORS Configuration
```python
CORS(app, supports_credentials=True)
# Allows cross-origin requests with cookies
```

### 4. Route Protection
```python
@app.route('/api/admin/audit-logs')
@admin_required  # Decorator checks session
def get_audit_logs():
    ...
```

### 5. File Upload Validation
```python
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'mp4', 'avi', 'mov'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
```

### 6. Audit Logging
```python
add_audit_log(
    user_id='admin',
    user_type='admin',
    action='enroll_faces',
    entity_type='student',
    entity_id=student_id,
    details=f'Enrolled {len(face_images)} faces'
)
```

---

## ⚡ Performance Optimizations

### 1. Frame Sampling
```python
sample_rate = 30  # Analyze every 30th frame
if frame_count % sample_rate == 0:
    analyze_frame()
```

### 2. Image Resizing
```python
# Standardize to 224x224 for faster inference
img = cv2.resize(img, (224, 224))
```

### 3. Lazy Model Loading
```python
# Load model once at startup, not per request
if __name__ == '__main__':
    load_model()
    app.run()
```

### 4. Database Connection Pooling
```python
# Close connections after each query
conn = sqlite3.connect('behavior_analysis.db')
# ... queries ...
conn.close()
```

### 5. Temporary File Cleanup
```python
# Delete uploaded files after processing
os.remove(filepath)
```

---

## 🐛 Error Handling

### API Errors
```python
try:
    result = predict_behavior(filepath)
    return jsonify({'success': True, 'result': result})
except Exception as e:
    print(f"Error: {e}")
    return jsonify({'error': str(e)}), 500
```

### Database Errors
```python
try:
    cursor.execute(query, params)
    conn.commit()
except sqlite3.IntegrityError:
    return False  # Duplicate entry
except Exception as e:
    print(f"Database error: {e}")
    return False
finally:
    conn.close()
```

### Model Loading Errors
```python
try:
    model = tf.keras.models.load_model(model_path)
except Exception as e:
    print(f"Model not found: {e}")
    model = None  # Use mock predictions
```

---

## 📊 Data Flow Diagrams

### Upload Analysis Flow
```
Frontend → POST /api/predict (file)
  ↓
Flask receives file
  ↓
Save to uploads/
  ↓
Check file type (image/video)
  ↓
If image: preprocess_image() → predict_behavior()
If video: process_video() → predict per frame
  ↓
Aggregate results
  ↓
Return JSON response
  ↓
Frontend displays results
```

### Live Analysis Flow
```
Frontend captures webcam frame
  ↓
Convert to base64
  ↓
POST /api/analyze-frame (base64 image)
  ↓
Backend decodes base64 → numpy array
  ↓
predict_from_frame() → Swin Transformer
  ↓
Return behavior + confidence
  ↓
Frontend updates UI (every 2 seconds)
```

### Classroom Analysis Flow
```
Upload classroom video
  ↓
POST /api/analyze-classroom-video
  ↓
For each sampled frame:
  YOLOv8n detects persons
    ↓
  For each person:
    InsightFace identifies student
    Swin Transformer predicts behavior
    ↓
  Aggregate by student_id
  ↓
Return per-student statistics
  ↓
Frontend displays individual reports
```

---

## 🧪 Testing

### Manual Testing
```bash
# Start server
python app.py

# Test endpoints with curl
curl -X POST http://localhost:5000/api/admin/login \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","password":"admin123"}'
```

### Unit Tests (Future)
```python
# tests/test_database.py
import unittest
from database import add_student, get_student

class TestDatabase(unittest.TestCase):
    def test_add_student(self):
        result = add_student('S999', 'Test Student')
        self.assertTrue(result)
```

---

## 🚀 Deployment

### Development
```bash
python app.py
# Runs on http://localhost:5000
# Debug mode enabled
```

### Production
```bash
# Use production WSGI server
pip install gunicorn

gunicorn -w 4 -b 0.0.0.0:5000 app:app
# 4 workers, bind to all interfaces
```

### Environment Variables
```bash
export ADMIN_USERNAME=admin
export ADMIN_PASSWORD_HASH=<hash>
export FLASK_ENV=production
export SECRET_KEY=<random-key>
```

---

## 📚 Dependencies

```txt
Flask>=2.3.0              # Web framework
flask-cors>=4.0.0         # CORS support
Werkzeug>=2.3.0           # WSGI utilities
tensorflow>=2.10.0        # ML framework
opencv-python>=4.5.0      # Video processing
ultralytics>=8.0.0        # YOLOv8
insightface>=0.7.3        # Face recognition
onnxruntime>=1.15.0       # ONNX runtime
numpy>=1.21.0             # Numerical computing
```

---

## 🔄 Future Improvements

1. **Upgrade to bcrypt** for password hashing
2. **Add PostgreSQL** for better concurrency
3. **Implement JWT tokens** instead of sessions
4. **Add rate limiting** to prevent abuse
5. **Use Redis** for session storage
6. **Add WebSocket** for real-time updates
7. **Implement caching** for frequent queries
8. **Add API versioning** (/api/v1/...)
9. **Create comprehensive tests** (unit + integration)
10. **Add Swagger/OpenAPI** documentation

---

**Next**: See [ALGORITHM_SETUP.md](ALGORITHM_SETUP.md) for detailed algorithm configuration.

