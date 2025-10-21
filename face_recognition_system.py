"""
Face Recognition System for Student Identification
Detects and identifies multiple students in classroom videos
"""

import cv2
import numpy as np
import os
import pickle
from pathlib import Path

# Face detection using Haar Cascade or DNN
class FaceDetector:
    def __init__(self, method='haar'):
        self.method = method
        
        if method == 'haar':
            # Haar Cascade (faster, less accurate)
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.detector = cv2.CascadeClassifier(cascade_path)
        
        elif method == 'dnn':
            # DNN face detector (more accurate)
            model_file = 'models/opencv_face_detector_uint8.pb'
            config_file = 'models/opencv_face_detector.pbtxt'
            
            if os.path.exists(model_file) and os.path.exists(config_file):
                self.detector = cv2.dnn.readNetFromTensorflow(model_file, config_file)
            else:
                print("DNN model files not found, falling back to Haar Cascade")
                cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                self.detector = cv2.CascadeClassifier(cascade_path)
                self.method = 'haar'
    
    def detect_faces(self, frame):
        """Detect faces in frame, return list of (x, y, w, h) boxes"""
        if self.method == 'haar':
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            return faces
        
        elif self.method == 'dnn':
            blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], False, False)
            self.detector.setInput(blob)
            detections = self.detector.forward()
            
            h, w = frame.shape[:2]
            faces = []
            
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.5:
                    x1 = int(detections[0, 0, i, 3] * w)
                    y1 = int(detections[0, 0, i, 4] * h)
                    x2 = int(detections[0, 0, i, 5] * w)
                    y2 = int(detections[0, 0, i, 6] * h)
                    faces.append([x1, y1, x2-x1, y2-y1])
            
            return np.array(faces)
        
        return []


class FaceRecognizer:
    """Simple face recognition using LBPH (Local Binary Patterns Histograms)"""
    
    def __init__(self, faces_dir='student_faces', encodings_file='face_encodings.pkl'):
        self.faces_dir = faces_dir
        self.encodings_file = encodings_file
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.known_faces = {}
        self.trained = False
        
        # Create faces directory if it doesn't exist
        os.makedirs(faces_dir, exist_ok=True)
        
        # Load existing encodings if available
        self.load_encodings()
    
    def enroll_student(self, student_id, name, face_images):
        """
        Enroll a student by providing their face images
        face_images: list of numpy arrays (face images)
        """
        student_dir = os.path.join(self.faces_dir, student_id)
        os.makedirs(student_dir, exist_ok=True)
        
        # Save face images
        for idx, img in enumerate(face_images):
            img_path = os.path.join(student_dir, f'face_{idx}.jpg')
            cv2.imwrite(img_path, img)
        
        self.known_faces[student_id] = {
            'name': name,
            'images': face_images
        }
        
        print(f"Enrolled student: {student_id} - {name} with {len(face_images)} images")
    
    def train(self):
        """Train the face recognizer on all enrolled students"""
        faces = []
        labels = []
        label_map = {}
        
        # Load all student faces
        for idx, student_id in enumerate(self.known_faces.keys()):
            label_map[idx] = student_id
            student_dir = os.path.join(self.faces_dir, student_id)
            
            if os.path.exists(student_dir):
                for img_file in os.listdir(student_dir):
                    if img_file.endswith(('.jpg', '.png')):
                        img_path = os.path.join(student_dir, img_file)
                        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                        if img is not None:
                            faces.append(img)
                            labels.append(idx)
        
        if len(faces) > 0:
            self.recognizer.train(faces, np.array(labels))
            self.trained = True
            self.label_map = label_map
            print(f"Training complete! Trained on {len(faces)} face images from {len(label_map)} students")
            
            # Save encodings
            self.save_encodings()
        else:
            print("No faces to train on")
    
    def identify_face(self, face_img, confidence_threshold=100):
        """
        Identify a face image
        Returns: (student_id, confidence) or (None, 0) if not recognized
        Lower confidence = better match
        """
        if not self.trained:
            return None, 0
        
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY) if len(face_img.shape) == 3 else face_img
        label, confidence = self.recognizer.predict(gray)
        
        if confidence < confidence_threshold:
            student_id = self.label_map.get(label)
            return student_id, confidence
        
        return None, confidence
    
    def save_encodings(self):
        """Save trained model and mappings"""
        data = {
            'known_faces': self.known_faces,
            'label_map': self.label_map if self.trained else {},
            'trained': self.trained
        }
        
        with open(self.encodings_file, 'wb') as f:
            pickle.dump(data, f)
        
        if self.trained:
            self.recognizer.save('face_recognizer_model.yml')
        
        print(f"Saved encodings to {self.encodings_file}")
    
    def load_encodings(self):
        """Load previously saved encodings"""
        if os.path.exists(self.encodings_file):
            try:
                with open(self.encodings_file, 'rb') as f:
                    data = pickle.load(f)
                
                self.known_faces = data.get('known_faces', {})
                self.label_map = data.get('label_map', {})
                self.trained = data.get('trained', False)
                
                if self.trained and os.path.exists('face_recognizer_model.yml'):
                    self.recognizer.read('face_recognizer_model.yml')
                    print(f"Loaded {len(self.known_faces)} enrolled students")
                
            except Exception as e:
                print(f"Error loading encodings: {e}")


class ClassroomAnalyzer:
    """Analyze full classroom video with multiple students"""
    
    def __init__(self, behavior_predictor_func):
        self.face_detector = FaceDetector(method='haar')
        self.face_recognizer = FaceRecognizer()
        self.behavior_predictor = behavior_predictor_func
        
    def analyze_frame(self, frame):
        """
        Analyze a classroom frame
        Returns: list of detected students with their behaviors
        [
            {
                'student_id': 'S001',
                'name': 'John Doe',
                'bbox': (x, y, w, h),
                'behavior': 'Reading',
                'confidence': 0.85
            },
            ...
        ]
        """
        results = []
        
        # Detect all faces
        faces = self.face_detector.detect_faces(frame)
        
        for (x, y, w, h) in faces:
            # Extract face region with some padding
            padding = 20
            y1 = max(0, y - padding)
            y2 = min(frame.shape[0], y + h + padding)
            x1 = max(0, x - padding)
            x2 = min(frame.shape[1], x + w + padding)
            
            face_img = frame[y1:y2, x1:x2]
            
            if face_img.size == 0:
                continue
            
            # Identify student
            student_id, conf = self.face_recognizer.identify_face(face_img)
            
            # Predict behavior for this region
            behavior, behavior_conf = self.behavior_predictor(face_img)
            
            result = {
                'student_id': student_id if student_id else 'Unknown',
                'name': self.face_recognizer.known_faces.get(student_id, {}).get('name', 'Unknown') if student_id else 'Unknown',
                'bbox': (x, y, w, h),
                'face_confidence': conf,
                'behavior': behavior,
                'behavior_confidence': behavior_conf
            }
            
            results.append(result)
        
        return results
    
    def process_video(self, video_path, sample_rate=30):
        """
        Process entire classroom video
        Returns: dict with per-student analysis
        """
        cap = cv2.VideoCapture(video_path)
        
        student_data = {}  # student_id -> {frames: [], behaviors: []}
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % sample_rate == 0:
                # Analyze this frame
                detections = self.analyze_frame(frame)
                
                for detection in detections:
                    student_id = detection['student_id']
                    
                    if student_id not in student_data:
                        student_data[student_id] = {
                            'name': detection['name'],
                            'frames': [],
                            'behaviors': {
                                'Raising Hand': 0,
                                'Reading': 0,
                                'Sleeping': 0,
                                'Writing': 0
                            }
                        }
                    
                    student_data[student_id]['frames'].append({
                        'frame': frame_count,
                        'behavior': detection['behavior'],
                        'confidence': detection['behavior_confidence']
                    })
                    
                    student_data[student_id]['behaviors'][detection['behavior']] += 1
            
            frame_count += 1
        
        cap.release()
        
        # Calculate statistics for each student
        for student_id in student_data:
            total = sum(student_data[student_id]['behaviors'].values())
            if total > 0:
                student_data[student_id]['behavior_stats'] = [
                    {
                        'label': behavior,
                        'count': count,
                        'percentage': (count / total) * 100
                    }
                    for behavior, count in student_data[student_id]['behaviors'].items()
                ]
        
        return student_data


def capture_face_samples(student_id, name, num_samples=10):
    """
    Capture face samples from webcam for enrollment
    """
    cap = cv2.VideoCapture(0)
    face_detector = FaceDetector()
    samples = []
    count = 0
    
    print(f"Capturing {num_samples} face samples for {name}...")
    print("Press SPACE to capture, ESC to cancel")
    
    while count < num_samples:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect face
        faces = face_detector.detect_faces(frame)
        
        # Draw rectangle around face
        display_frame = frame.copy()
        for (x, y, w, h) in faces:
            cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        cv2.putText(display_frame, f"Captured: {count}/{num_samples}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(display_frame, "Press SPACE to capture", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow('Capture Face Samples', display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == 27:  # ESC
            print("Cancelled")
            break
        
        if key == 32 and len(faces) > 0:  # SPACE
            # Capture the first detected face
            x, y, w, h = faces[0]
            face_img = frame[y:y+h, x:x+w]
            
            # Resize to standard size
            face_img = cv2.resize(face_img, (200, 200))
            samples.append(face_img)
            count += 1
            print(f"Captured sample {count}/{num_samples}")
    
    cap.release()
    cv2.destroyAllWindows()
    
    return samples


if __name__ == '__main__':
    # Example usage: Enroll a student
    print("Face Recognition System Setup")
    print("=" * 50)
    
    recognizer = FaceRecognizer()
    
    # Check if we need to enroll students
    if len(recognizer.known_faces) == 0:
        print("\nNo students enrolled. Let's enroll some students!")
        
        while True:
            choice = input("\nEnroll a student? (y/n): ").lower()
            if choice != 'y':
                break
            
            student_id = input("Enter Student ID (e.g., S001): ")
            name = input("Enter Student Name: ")
            
            # Capture face samples
            samples = capture_face_samples(student_id, name, num_samples=10)
            
            if len(samples) > 0:
                recognizer.enroll_student(student_id, name, samples)
            else:
                print("No samples captured, skipping...")
        
        # Train the recognizer
        if len(recognizer.known_faces) > 0:
            print("\nTraining face recognizer...")
            recognizer.train()
    else:
        print(f"\n{len(recognizer.known_faces)} students already enrolled")
        print("Run with --enroll to add more students")


