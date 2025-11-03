from flask import Flask, request, jsonify, send_file, session
from flask_cors import CORS
import os
import cv2
import numpy as np
from werkzeug.utils import secure_filename
import base64
from pathlib import Path
import time
from datetime import datetime
import zipfile
import shutil
from database import (
    init_db, add_student, get_student, save_report,
    get_student_reports, get_report_details, get_all_students,
    verify_student_password, add_audit_log, get_audit_logs
)
from face_recognition_system import (
    FaceDetector, FaceRecognizer, ClassroomAnalyzer, capture_face_samples
)
from admin_auth import (
    login_admin, logout_admin, is_admin_logged_in, admin_required, change_admin_password
)

# Performance optimization
import torch
torch.set_num_threads(16)  # Use all CPU threads (Ryzen 7 5700U)
torch.set_num_interop_threads(4)  # Parallel operations
os.environ['OMP_NUM_THREADS'] = '16'
os.environ['MKL_NUM_THREADS'] = '16'

app = Flask(__name__)
CORS(app, supports_credentials=True)  # Enable CORS with credentials for session
app.secret_key = os.urandom(24)  # Secret key for sessions
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif', 'mp4', 'avi', 'mov'}

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables
model = None
device = None
LABELS = ['Looking_Forward', 'Raising_Hand', 'Reading', 'Sleeping', 'Standing', 'Turning_Around', 'Writting']

# Face recognition system
face_recognizer = None
classroom_analyzer = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def load_model():
    """Load the trained PyTorch behavior model"""
    global model, device, face_recognizer, classroom_analyzer
    
    try:
        import torch
        import torch.nn as nn
        import timm
        
        # Define BehaviorClassifier (matches training code)
        class BehaviorClassifier(nn.Module):
            def __init__(self, num_classes=7, pretrained=False, model_name='swin_tiny_patch4_window7_224'):
                super().__init__()
                self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
                self.num_classes = num_classes
            
            def forward(self, x):
                return self.backbone(x)
        
        # Check for GPU
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # Load model checkpoint
        model_path = './temptrainedoutput/best_behavior_model_fixed.pth'
        if not os.path.exists(model_path):
            print(f"⚠ Model file not found at {model_path}")
            model = None
        else:
            checkpoint = torch.load(model_path, map_location=device)
            
            # Create model
            model = BehaviorClassifier(num_classes=7, pretrained=False)
            
            # Load state dict
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            
            model.to(device)
            model.eval()
            print(f"✓ PyTorch behavior model loaded successfully from {model_path}")
            
    except Exception as e:
        print(f"⚠ Error loading behavior model: {e}")
        import traceback
        traceback.print_exc()
        model = None
        device = torch.device('cpu')
    
    # Initialize face recognition system
    try:
        face_recognizer = FaceRecognizer()
        classroom_analyzer = ClassroomAnalyzer(predict_from_frame)
        print("✓ Face recognition system initialized!")
    except Exception as e:
        print(f"⚠ Error initializing face recognition: {e}")
        face_recognizer = None
        classroom_analyzer = None

def preprocess_image(image_path):
    """Preprocess image for model prediction"""
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def predict_behavior(image_path):
    """Predict student behavior from image using PyTorch"""
    if model is None:
        # Return mock predictions if model is not loaded
        import random
        behaviors = ['Raising Hand', 'Reading', 'Writing', 'Sleeping']
        top_idx = random.randint(0, 3)
        confidences = [0.1, 0.15, 0.2, 0.55]
        random.shuffle(confidences)
        
        return {
            'predictions': [
                {'label': behaviors[i], 'confidence': confidences[i]}
                for i in range(4)
            ],
            'top_prediction': behaviors[top_idx],
            'confidence': confidences[top_idx]
        }
    
    try:
        import torch
        from torchvision import transforms
        from PIL import Image
        
        # Load and transform image
        img = Image.open(image_path).convert('RGB')
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        # Predict
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.softmax(outputs, dim=1)[0]
        
        # Sort predictions by confidence
        pred_list = [
            {'label': LABELS[i], 'confidence': float(probs[i].item())}
            for i in range(len(LABELS))
        ]
        pred_list.sort(key=lambda x: x['confidence'], reverse=True)
        
        return {
            'predictions': pred_list,
            'top_prediction': pred_list[0]['label'],
            'confidence': pred_list[0]['confidence']
        }
    except Exception as e:
        print(f"Error during prediction: {e}")
        import traceback
        traceback.print_exc()
        return None

def predict_from_frame(frame):
    """Predict behavior from a numpy frame using PyTorch"""
    if model is None:
        import random
        behaviors = ['Raising Hand', 'Reading', 'Writing', 'Sleeping']
        top_idx = random.randint(0, 3)
        confidences = [0.1, 0.15, 0.2, 0.55]
        random.shuffle(confidences)
        
        return behaviors[top_idx], confidences[top_idx]
    
    try:
        import torch
        from torchvision import transforms
        from PIL import Image
        
        # Convert BGR to RGB
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        
        # Apply transforms (same as training)
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        # Predict
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.softmax(outputs, dim=1)[0]
            top_idx = torch.argmax(probs).item()
            confidence = probs[top_idx].item()
        
        return LABELS[top_idx], float(confidence)
    except Exception as e:
        print(f"Error during frame prediction: {e}")
        import traceback
        traceback.print_exc()
        return 'Unknown', 0.0

def process_video(video_path):
    """Process video and predict behaviors for frames"""
    cap = cv2.VideoCapture(video_path)
    predictions = []
    frame_count = 0
    sample_rate = 10  # Analyze every 10th frame
    
    while cap.isOpened() and frame_count < 100:  # Limit to 100 frames
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % sample_rate == 0:
            # Save temporary frame
            temp_path = os.path.join(app.config['UPLOAD_FOLDER'], f'temp_frame_{frame_count}.jpg')
            cv2.imwrite(temp_path, frame)
            
            # Predict
            result = predict_behavior(temp_path)
            if result:
                predictions.append({
                    'frame': frame_count,
                    'prediction': result['top_prediction'],
                    'confidence': result['confidence']
                })
            
            # Clean up temp file
            os.remove(temp_path)
        
        frame_count += 1
    
    cap.release()
    
    # Aggregate predictions
    if predictions:
        behavior_counts = {}
        for pred in predictions:
            behavior = pred['prediction']
            behavior_counts[behavior] = behavior_counts.get(behavior, 0) + 1
        
        total = len(predictions)
        behavior_stats = [
            {
                'label': behavior,
                'count': count,
                'percentage': (count / total) * 100
            }
            for behavior, count in behavior_counts.items()
        ]
        behavior_stats.sort(key=lambda x: x['count'], reverse=True)
        
        return {
            'total_frames': frame_count,
            'analyzed_frames': len(predictions),
            'behavior_stats': behavior_stats,
            'frame_predictions': predictions
        }
    
    return None

@app.route('/api/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Check if it's a video or image
        file_ext = filename.rsplit('.', 1)[1].lower()
        
        if file_ext in {'mp4', 'avi', 'mov'}:
            # Process video with classroom analyzer (includes face recognition)
            if classroom_analyzer:
                student_data = classroom_analyzer.process_video(filepath, sample_rate=10)
                result = {
                    'student_data': student_data,
                    'total_students': len(student_data) if student_data else 0
                }
            else:
                # Fallback to simple behavior classification
                result = process_video(filepath)
            result_type = 'video'
        else:
            # Process image
            result = predict_behavior(filepath)
            result_type = 'image'
        
        # Clean up uploaded file
        # os.remove(filepath)  # Commented out to keep files for debugging
        
        if result:
            return jsonify({
                'success': True,
                'type': result_type,
                'result': result
            })
        else:
            return jsonify({'error': 'Prediction failed'}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get dataset statistics"""
    stats = {
        'total_samples': 720,
        'train_samples': 460,
        'validation_samples': 116,
        'test_samples': 144,
        'classes': LABELS,
        'class_distribution': [
            {'label': 'Raising Hand', 'count': 180, 'percentage': 25},
            {'label': 'Reading', 'count': 180, 'percentage': 25},
            {'label': 'Sleeping', 'count': 180, 'percentage': 25},
            {'label': 'Writing', 'count': 180, 'percentage': 25}
        ]
    }
    return jsonify(stats)

@app.route('/api/model-info', methods=['GET'])
def model_info():
    """Get model information"""
    info = {
        'model_loaded': model is not None,
        'architecture': 'Swin Transformer',
        'input_size': [224, 224, 3],
        'num_classes': 4,
        'classes': LABELS
    }
    return jsonify(info)

# ============= STUDENT MANAGEMENT =============

@app.route('/api/students', methods=['GET'])
def list_students():
    """Get all students"""
    students = get_all_students()
    return jsonify({'students': students})

@app.route('/api/students/<student_id>', methods=['GET'])
def get_student_info(student_id):
    """Get student information"""
    student = get_student(student_id)
    if student:
        return jsonify(student)
    return jsonify({'error': 'Student not found'}), 404

@app.route('/api/students', methods=['POST'])
def create_student():
    """Create a new student"""
    data = request.json
    student_id = data.get('student_id')
    name = data.get('name')
    email = data.get('email')
    password = data.get('password')
    
    if not student_id or not name:
        return jsonify({'error': 'student_id and name are required'}), 400
    
    success = add_student(student_id, name, email, password)
    if success:
        # Log the action
        user_id = session.get('admin_logged_in') and 'admin' or student_id
        add_audit_log(user_id, 'admin', 'create_student', 'student', student_id, f'Created student {name}')
        return jsonify({'message': 'Student created successfully', 'student_id': student_id})
    return jsonify({'error': 'Student already exists'}), 400

# ============= STUDENT AUTHENTICATION =============

@app.route('/api/student/login', methods=['POST'])
def student_login():
    """Student login"""
    try:
        data = request.json
        student_id = data.get('student_id')
        password = data.get('password')
        
        if not student_id or not password:
            return jsonify({'error': 'Student ID and password required'}), 400
        
        if verify_student_password(student_id, password):
            session['student_logged_in'] = True
            session['student_id'] = student_id
            
            # Get student info
            student = get_student(student_id)
            
            # Log the action
            add_audit_log(student_id, 'student', 'login', None, None, 'Student logged in')
            
            return jsonify({
                'success': True,
                'student': student
            })
        else:
            return jsonify({'error': 'Invalid credentials'}), 401
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/student/logout', methods=['POST'])
def student_logout():
    """Student logout"""
    student_id = session.get('student_id')
    if student_id:
        add_audit_log(student_id, 'student', 'logout', None, None, 'Student logged out')
    
    session.pop('student_logged_in', None)
    session.pop('student_id', None)
    return jsonify({'success': True})

@app.route('/api/student/status', methods=['GET'])
def student_status():
    """Check student login status"""
    return jsonify({
        'logged_in': session.get('student_logged_in', False),
        'student_id': session.get('student_id')
    })

# ============= LIVE WEBCAM ANALYSIS =============

@app.route('/api/analyze-frame', methods=['POST'])
def analyze_frame():
    """Analyze a single frame from webcam"""
    try:
        data = request.json
        image_data = data.get('frame')
        
        if not image_data:
            return jsonify({'error': 'No frame data'}), 400
        
        # Decode base64 image
        if 'base64,' in image_data:
            image_data = image_data.split('base64,')[1]
        
        img_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Predict
        behavior, confidence = predict_from_frame(frame)
        
        return jsonify({
            'success': True,
            'behavior': behavior,
            'confidence': float(confidence)
        })
        
    except Exception as e:
        print(f"Error analyzing frame: {e}")
        return jsonify({'error': str(e)}), 500

# ============= REPORT MANAGEMENT =============

@app.route('/api/reports/save', methods=['POST'])
def save_analysis_report():
    """Save a complete analysis session report"""
    try:
        data = request.json
        student_id = data.get('student_id')
        duration = data.get('duration', 0)
        behavior_stats = data.get('behavior_stats', [])
        frame_data = data.get('frame_data', [])
        notes = data.get('notes', '')
        
        if not student_id:
            return jsonify({'error': 'student_id is required'}), 400
        
        # Check if student exists
        student = get_student(student_id)
        if not student:
            return jsonify({'error': 'Student not found'}), 404
        
        # Save report
        report_id = save_report(student_id, duration, behavior_stats, frame_data, notes)
        
        # Log the action
        user_id = session.get('student_id') or session.get('admin_logged_in') and 'admin' or 'system'
        user_type = 'student' if session.get('student_id') else 'admin'
        add_audit_log(user_id, user_type, 'create_report', 'report', str(report_id), f'Created report for {student_id}')
        
        return jsonify({
            'success': True,
            'report_id': report_id,
            'message': 'Report saved successfully'
        })
        
    except Exception as e:
        print(f"Error saving report: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/reports/student/<student_id>', methods=['GET'])
def get_student_report_list(student_id):
    """Get all reports for a student"""
    try:
        reports = get_student_reports(student_id)
        return jsonify({'reports': reports})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/reports/<int:report_id>', methods=['GET'])
def get_single_report(report_id):
    """Get detailed report"""
    try:
        report = get_report_details(report_id)
        if report:
            return jsonify(report)
        return jsonify({'error': 'Report not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ============= ADMIN AUTHENTICATION =============

@app.route('/api/admin/login', methods=['POST'])
def admin_login():
    """Admin login"""
    try:
        data = request.json
        username = data.get('username')
        password = data.get('password')
        
        if not username or not password:
            return jsonify({'error': 'Username and password required'}), 400
        
        success, message = login_admin(username, password)
        
        if success:
            return jsonify({
                'success': True,
                'message': message
            })
        else:
            return jsonify({'error': message}), 401
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/admin/logout', methods=['POST'])
def admin_logout():
    """Admin logout"""
    logout_admin()
    return jsonify({'success': True, 'message': 'Logged out successfully'})

@app.route('/api/admin/status', methods=['GET'])
def admin_status():
    """Check admin login status"""
    return jsonify({
        'logged_in': is_admin_logged_in()
    })

@app.route('/api/admin/change-password', methods=['POST'])
@admin_required
def admin_change_password():
    """Change admin password"""
    try:
        data = request.json
        old_password = data.get('old_password')
        new_password = data.get('new_password')
        
        if not old_password or not new_password:
            return jsonify({'error': 'Old and new passwords required'}), 400
        
        success, message = change_admin_password(old_password, new_password)
        
        if success:
            add_audit_log('admin', 'admin', 'change_password', None, None, 'Admin changed password')
            return jsonify({'success': True, 'message': message})
        else:
            return jsonify({'error': message}), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/admin/audit-logs', methods=['GET'])
@admin_required
def get_admin_audit_logs():
    """Get audit logs (ADMIN ONLY)"""
    try:
        limit = request.args.get('limit', 100, type=int)
        logs = get_audit_logs(limit)
        return jsonify({'logs': logs})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ============= DOWNLOAD REPORTS =============

@app.route('/api/reports/<int:report_id>/download', methods=['GET'])
def download_report(report_id):
    """Download report as JSON file"""
    try:
        report = get_report_details(report_id)
        if not report:
            return jsonify({'error': 'Report not found'}), 404
        
        # Create filename with student ID and date
        student_id = report.get('student_id', 'Unknown')
        date_str = report.get('session_date', '').split('T')[0]
        filename = f"report_{student_id}_{date_str}.json"
        
        # Create response with file content
        import json
        response = jsonify(report)
        response.headers['Content-Disposition'] = f'attachment; filename={filename}'
        response.headers['Content-Type'] = 'application/json'
        
        return response
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/reports/download-multiple', methods=['POST'])
def download_multiple_reports():
    """Download multiple reports as ZIP"""
    try:
        data = request.json
        report_ids = data.get('report_ids', [])
        
        if not report_ids:
            return jsonify({'error': 'No report IDs provided'}), 400
        
        # Create temporary ZIP file
        import tempfile
        import json
        
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.zip')
        temp_file.close()
        
        with zipfile.ZipFile(temp_file.name, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for report_id in report_ids:
                report = get_report_details(report_id)
                if report:
                    student_id = report.get('student_id', 'Unknown')
                    date_str = report.get('session_date', '').split('T')[0]
                    filename_in_zip = f"{student_id}/report_{date_str}.json"
                    
                    # Convert report to JSON string
                    report_json = json.dumps(report, indent=2, default=str)
                    zipf.writestr(filename_in_zip, report_json)
        
        # Send ZIP file
        return send_file(
            temp_file.name,
            as_attachment=True,
            download_name='student_reports.zip',
            mimetype='application/zip'
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ============= FACE RECOGNITION & ENROLLMENT =============

@app.route('/api/face/enroll', methods=['POST'])
@admin_required
def enroll_student_face():
    """Enroll student face samples (ADMIN ONLY)"""
    try:
        data = request.json
        student_id = data.get('student_id')
        name = data.get('name')
        face_images_b64 = data.get('face_images', [])  # Base64 encoded images
        
        if not student_id or not name or not face_images_b64:
            return jsonify({'error': 'student_id, name, and face_images required'}), 400
        
        # Decode face images
        face_images = []
        for img_b64 in face_images_b64:
            if 'base64,' in img_b64:
                img_b64 = img_b64.split('base64,')[1]
            
            img_bytes = base64.b64decode(img_b64)
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is not None:
                face_images.append(img)
        
        if len(face_images) == 0:
            return jsonify({'error': 'No valid face images provided'}), 400
        
        # Enroll in face recognition system
        if face_recognizer:
            face_recognizer.enroll_student(student_id, name, face_images)
            face_recognizer.train()
            
            # Log the action
            add_audit_log('admin', 'admin', 'enroll_faces', 'student', student_id, f'Enrolled {len(face_images)} faces for {name}')
            
            return jsonify({
                'success': True,
                'message': f'Enrolled {len(face_images)} face samples for {name}',
                'student_id': student_id
            })
        else:
            return jsonify({'error': 'Face recognition system not available'}), 500
            
    except Exception as e:
        print(f"Error enrolling face: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/face/enrolled', methods=['GET'])
def get_enrolled_faces():
    """Get list of enrolled students"""
    try:
        if face_recognizer:
            enrolled = [
                {'student_id': sid, 'name': data.get('name', 'Unknown')}
                for sid, data in face_recognizer.known_faces.items()
            ]
            return jsonify({
                'enrolled_students': enrolled,
                'count': len(enrolled),
                'trained': face_recognizer.trained
            })
        return jsonify({'error': 'Face recognition system not available'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ============= CLASSROOM VIDEO ANALYSIS =============

@app.route('/api/analyze-classroom-frame', methods=['POST'])
def analyze_classroom_frame():
    """Analyze classroom frame with multiple students"""
    try:
        data = request.json
        image_data = data.get('frame')
        
        if not image_data:
            return jsonify({'error': 'No frame data'}), 400
        
        # Decode base64 image
        if 'base64,' in image_data:
            image_data = image_data.split('base64,')[1]
        
        img_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if classroom_analyzer:
            # Analyze frame and detect all students
            detections = classroom_analyzer.analyze_frame(frame)
            
            return jsonify({
                'success': True,
                'detections': detections,
                'count': len(detections)
            })
        else:
            return jsonify({'error': 'Classroom analyzer not available'}), 500
        
    except Exception as e:
        print(f"Error analyzing classroom frame: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/analyze-classroom-video', methods=['POST'])
def analyze_classroom_video():
    """Analyze uploaded classroom video with face recognition"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            if classroom_analyzer:
                # Process video and get per-student analysis
                student_data = classroom_analyzer.process_video(filepath, sample_rate=30)
                
                # Clean up
                # os.remove(filepath)
                
                return jsonify({
                    'success': True,
                    'student_data': student_data,
                    'total_students': len(student_data)
                })
            else:
                return jsonify({'error': 'Classroom analyzer not available'}), 500
        
        return jsonify({'error': 'Invalid file'}), 400
        
    except Exception as e:
        print(f"Error analyzing classroom video: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/face/status', methods=['GET'])
def face_recognition_status():
    """Get face recognition system status"""
    try:
        if face_recognizer:
            return jsonify({
                'available': True,
                'enrolled_count': len(face_recognizer.known_faces),
                'trained': face_recognizer.trained,
                'enrolled_students': list(face_recognizer.known_faces.keys())
            })
        return jsonify({
            'available': False,
            'enrolled_count': 0,
            'trained': False
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ============= BULK FACE UPLOAD (ADMIN ONLY) =============

@app.route('/api/admin/bulk-upload-faces', methods=['POST'])
@admin_required
def bulk_upload_faces():
    """
    Upload face dataset in bulk (ADMIN ONLY)
    Accepts ZIP file with structure:
    faces.zip/
      ├── S001_JohnDoe/
      │   ├── face1.jpg
      │   ├── face2.jpg
      │   └── ...
      ├── S002_JaneSmith/
      └── ...
    """
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not file.filename.endswith('.zip'):
            return jsonify({'error': 'Only ZIP files are accepted'}), 400
        
        # Save uploaded file
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], 'faces_upload.zip')
        file.save(upload_path)
        
        # Extract ZIP
        extract_path = os.path.join(app.config['UPLOAD_FOLDER'], 'extracted_faces')
        os.makedirs(extract_path, exist_ok=True)
        
        with zipfile.ZipFile(upload_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        
        # Process extracted folders
        enrolled_count = 0
        errors = []
        
        for folder_name in os.listdir(extract_path):
            folder_path = os.path.join(extract_path, folder_name)
            
            if not os.path.isdir(folder_path):
                continue
            
            # Parse folder name: S001_JohnDoe or S001
            parts = folder_name.split('_')
            if len(parts) >= 2:
                student_id = parts[0]
                name = '_'.join(parts[1:])
            else:
                student_id = folder_name
                name = folder_name
            
            # Load face images from folder
            face_images = []
            for img_file in os.listdir(folder_path):
                img_path = os.path.join(folder_path, img_file)
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img = cv2.imread(img_path)
                    if img is not None:
                        # Resize to standard size
                        img = cv2.resize(img, (200, 200))
                        face_images.append(img)
            
            if len(face_images) > 0:
                try:
                    # Create student in database if doesn't exist
                    add_student(student_id, name)
                    
                    # Enroll faces
                    if face_recognizer:
                        face_recognizer.enroll_student(student_id, name, face_images)
                        enrolled_count += 1
                except Exception as e:
                    errors.append(f"{student_id}: {str(e)}")
        
        # Train the recognizer
        if face_recognizer and enrolled_count > 0:
            face_recognizer.train()
        
        # Cleanup
        os.remove(upload_path)
        shutil.rmtree(extract_path)
        
        return jsonify({
            'success': True,
            'enrolled_count': enrolled_count,
            'errors': errors,
            'message': f'Successfully enrolled {enrolled_count} students'
        })
        
    except Exception as e:
        print(f"Error in bulk upload: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/admin/upload-student-faces', methods=['POST'])
@admin_required
def upload_student_faces():
    """
    Upload face images for a specific student (ADMIN ONLY)
    Multipart form with student_id, name, and multiple image files
    """
    try:
        student_id = request.form.get('student_id')
        name = request.form.get('name')
        
        if not student_id or not name:
            return jsonify({'error': 'student_id and name required'}), 400
        
        # Get uploaded files
        face_images = []
        for key in request.files:
            file = request.files[key]
            if file and file.filename:
                # Read image
                img_bytes = file.read()
                nparr = np.frombuffer(img_bytes, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if img is not None:
                    img = cv2.resize(img, (200, 200))
                    face_images.append(img)
        
        if len(face_images) == 0:
            return jsonify({'error': 'No valid face images provided'}), 400
        
        # Create student in database
        add_student(student_id, name)
        
        # Enroll faces
        if face_recognizer:
            face_recognizer.enroll_student(student_id, name, face_images)
            face_recognizer.train()
            
            return jsonify({
                'success': True,
                'message': f'Enrolled {len(face_images)} faces for {name}',
                'student_id': student_id
            })
        else:
            return jsonify({'error': 'Face recognizer not available'}), 500
            
    except Exception as e:
        print(f"Error uploading student faces: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/admin/delete-student-faces/<student_id>', methods=['DELETE'])
@admin_required
def delete_student_faces(student_id):
    """Delete student's face data (ADMIN ONLY)"""
    try:
        if face_recognizer:
            # Remove from face recognizer
            if student_id in face_recognizer.known_faces:
                del face_recognizer.known_faces[student_id]
                
                # Delete face images folder
                face_folder = os.path.join('student_faces', student_id)
                if os.path.exists(face_folder):
                    shutil.rmtree(face_folder)
                
                # Retrain
                face_recognizer.train()
                
                return jsonify({
                    'success': True,
                    'message': f'Deleted face data for {student_id}'
                })
            else:
                return jsonify({'error': 'Student not found in face database'}), 404
        else:
            return jsonify({'error': 'Face recognizer not available'}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting Student Behavior Analysis System...")
    print("Initializing database...")
    init_db()
    print("Loading model...")
    load_model()
    print("Server ready!")
    app.run(debug=True, port=5000, load_dotenv=False)

