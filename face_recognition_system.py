"""
Face Recognition System for Student Identification
Uses YOLOv8n for person/face detection and InsightFace Buffalo_l for recognition
"""

import cv2
import numpy as np
import os
import pickle
from pathlib import Path
from ultralytics import YOLO
import insightface
from insightface.app import FaceAnalysis

class YOLODetector:
    """YOLOv8n-based person and face detection"""
    
    def __init__(self, model_size='n', conf_threshold=0.5):
        """
        Initialize YOLO detector
        model_size: 'n' (nano), 's' (small), 'm' (medium), 'l' (large), 'x' (xlarge)
        """
        self.conf_threshold = conf_threshold
        print(f"Loading YOLOv8{model_size} model...")
        
        # Load YOLOv8 model (auto-downloads if not present)
        self.model = YOLO(f'yolov8{model_size}.pt')
        print("✓ YOLOv8 model loaded successfully!")
    
    def detect_persons(self, frame):
        """
        Detect persons in frame using YOLO
        Returns: list of bounding boxes [(x1, y1, x2, y2, confidence), ...]
        """
        results = self.model(frame, conf=self.conf_threshold, classes=[0])  # class 0 = person
        
        persons = []
        if results and len(results) > 0:
            boxes = results[0].boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())
                    persons.append((int(x1), int(y1), int(x2), int(y2), conf))
        
        return persons
    
    def detect_faces_yolo(self, frame):
        """
        Detect faces using YOLO (requires custom face detection model)
        For now, falls back to person detection
        """
        # If you have a custom YOLO face detection model, use it here
        # For now, detect persons and use upper portion as face region
        persons = self.detect_persons(frame)
        
        faces = []
        for (x1, y1, x2, y2, conf) in persons:
            # Estimate face region (upper 40% of person bbox)
            h = y2 - y1
            face_h = int(h * 0.4)
            face_box = (x1, y1, x2, y1 + face_h, conf)
            faces.append(face_box)
        
        return faces


class BuffaloFaceRecognizer:
    """
    Face recognition using InsightFace Buffalo_l model
    High accuracy deep learning-based face recognition
    """
    
    def __init__(self, faces_dir='student_faces', encodings_file='face_encodings_buffalo.pkl'):
        self.faces_dir = faces_dir
        self.encodings_file = encodings_file
        self.known_faces = {}
        self.trained = False
        
        # Create faces directory if it doesn't exist
        os.makedirs(faces_dir, exist_ok=True)
        
        print("Initializing InsightFace Buffalo_l model...")
        
        # Initialize FaceAnalysis with Buffalo_l model
        self.app = FaceAnalysis(
            name='buffalo_l',
            providers=['CPUExecutionProvider']  # Use 'CUDAExecutionProvider' if GPU available
        )
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        
        print("✓ InsightFace Buffalo_l model loaded successfully!")
        
        # Load existing encodings if available
        self.load_encodings()
    
    def get_face_embedding(self, face_img):
        """
        Extract face embedding using Buffalo_l
        Returns: 512-dimensional embedding vector
        """
        # Analyze face
        faces = self.app.get(face_img)
        
        if len(faces) > 0:
            # Return embedding of first detected face
            return faces[0].embedding
        
        return None
    
    def enroll_student(self, student_id, name, face_images):
        """
        Enroll a student by extracting embeddings from their face images
        face_images: list of numpy arrays (BGR images)
        """
        student_dir = os.path.join(self.faces_dir, student_id)
        os.makedirs(student_dir, exist_ok=True)
        
        embeddings = []
        
        # Extract embeddings from all face images
        for idx, img in enumerate(face_images):
            # Save image
            img_path = os.path.join(student_dir, f'face_{idx}.jpg')
            cv2.imwrite(img_path, img)
            
            # Extract embedding
            embedding = self.get_face_embedding(img)
            if embedding is not None:
                embeddings.append(embedding)
        
        if len(embeddings) > 0:
            # Store average embedding for this student
            avg_embedding = np.mean(embeddings, axis=0)
            
            self.known_faces[student_id] = {
                'name': name,
                'embedding': avg_embedding,
                'num_samples': len(embeddings)
            }
            
            self.trained = True
            self.save_encodings()
            
            print(f"✓ Enrolled {name} ({student_id}) with {len(embeddings)} face samples")
        else:
            print(f"⚠ No valid faces detected for {name}")
    
    def identify_face(self, face_img, similarity_threshold=0.4):
        """
        Identify a face using cosine similarity
        Returns: (student_id, similarity_score) or (None, 0)
        Higher similarity = better match (range: 0-1)
        """
        if not self.trained or len(self.known_faces) == 0:
            return None, 0
        
        # Get embedding for query face
        query_embedding = self.get_face_embedding(face_img)
        
        if query_embedding is None:
            return None, 0
        
        # Find best match using cosine similarity
        best_match = None
        best_similarity = 0
        
        for student_id, data in self.known_faces.items():
            stored_embedding = data['embedding']
            
            # Cosine similarity
            similarity = np.dot(query_embedding, stored_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(stored_embedding)
            )
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = student_id
        
        # Return match if above threshold
        if best_similarity >= similarity_threshold:
            return best_match, float(best_similarity)
        
        return None, float(best_similarity)
    
    def save_encodings(self):
        """Save face encodings to pickle file"""
        data = {
            'known_faces': self.known_faces,
            'trained': self.trained
        }
        
        with open(self.encodings_file, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"✓ Saved face encodings to {self.encodings_file}")
    
    def load_encodings(self):
        """Load previously saved face encodings"""
        if os.path.exists(self.encodings_file):
            try:
                with open(self.encodings_file, 'rb') as f:
                    data = pickle.load(f)
                
                self.known_faces = data.get('known_faces', {})
                self.trained = data.get('trained', False)
                
                print(f"✓ Loaded {len(self.known_faces)} enrolled students from {self.encodings_file}")
                
            except Exception as e:
                print(f"⚠ Error loading encodings: {e}")


class ClassroomAnalyzer:
    """Analyze full classroom video with YOLOv8 + InsightFace Buffalo_l"""
    
    def __init__(self, behavior_predictor_func):
        print("Initializing Classroom Analyzer...")
        self.yolo_detector = YOLODetector(model_size='n')  # YOLOv8n (fastest)
        self.face_recognizer = BuffaloFaceRecognizer()
        self.behavior_predictor = behavior_predictor_func
        print("✓ Classroom Analyzer ready!")
        
    def analyze_frame(self, frame):
        """
        Analyze a classroom frame with YOLO + InsightFace
        Returns: list of detected students with their behaviors
        """
        results = []
        
        # Detect all persons using YOLO
        persons = self.yolo_detector.detect_persons(frame)
        
        for (x1, y1, x2, y2, conf) in persons:
            # Extract person region with padding
            padding = 20
            h, w = frame.shape[:2]
            y1_pad = max(0, y1 - padding)
            y2_pad = min(h, y2 + padding)
            x1_pad = max(0, x1 - padding)
            x2_pad = min(w, x2 + padding)
            
            person_img = frame[y1_pad:y2_pad, x1_pad:x2_pad]
            
            if person_img.size == 0:
                continue
            
            # Extract face region (upper 40% of person)
            person_h = y2_pad - y1_pad
            face_region = person_img[0:int(person_h * 0.4), :]
            
            # Identify student using InsightFace
            student_id, similarity = self.face_recognizer.identify_face(face_region)
            
            # Predict behavior for full person region
            behavior, behavior_conf = self.behavior_predictor(person_img)
            
            result = {
                'student_id': student_id if student_id else 'Unknown',
                'name': self.face_recognizer.known_faces.get(student_id, {}).get('name', 'Unknown') if student_id else 'Unknown',
                'bbox': (x1, y1, x2, y2),
                'detection_confidence': conf,
                'face_similarity': similarity,
                'behavior': behavior,
                'behavior_confidence': behavior_conf
            }
            
            results.append(result)
        
        return results
    
    def process_video(self, video_path, sample_rate=30):
        """
        Process entire classroom video with YOLO + InsightFace
        Returns: dict with per-student analysis
        """
        cap = cv2.VideoCapture(video_path)
        
        student_data = {}  # student_id -> {frames: [], behaviors: []}
        frame_count = 0
        
        print(f"Processing video: {video_path}")
        
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
            
            # Progress indicator
            if frame_count % 100 == 0:
                print(f"  Processed {frame_count} frames...")
        
        cap.release()
        print(f"✓ Video processing complete! Total frames: {frame_count}")
        
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


# Legacy compatibility classes (for backward compatibility)
class FaceDetector:
    """Legacy face detector - now uses YOLO"""
    def __init__(self, method='yolo'):
        self.detector = YOLODetector(model_size='n')
    
    def detect_faces(self, frame):
        """Detect faces using YOLO, return in (x, y, w, h) format"""
        persons = self.detector.detect_persons(frame)
        
        faces = []
        for (x1, y1, x2, y2, conf) in persons:
            # Convert to (x, y, w, h) format
            w = x2 - x1
            h = y2 - y1
            # Face is upper 40%
            face_h = int(h * 0.4)
            faces.append([x1, y1, w, face_h])
        
        return np.array(faces)


class FaceRecognizer:
    """Legacy face recognizer - now uses InsightFace Buffalo_l"""
    def __init__(self, faces_dir='student_faces', encodings_file='face_encodings.pkl'):
        self.recognizer = BuffaloFaceRecognizer(faces_dir, encodings_file)
        self.known_faces = self.recognizer.known_faces
        self.trained = self.recognizer.trained
    
    def enroll_student(self, student_id, name, face_images):
        self.recognizer.enroll_student(student_id, name, face_images)
        self.known_faces = self.recognizer.known_faces
        self.trained = self.recognizer.trained
    
    def train(self):
        # Training happens automatically in Buffalo_l
        self.trained = self.recognizer.trained
    
    def identify_face(self, face_img, confidence_threshold=0.4):
        student_id, similarity = self.recognizer.identify_face(face_img, confidence_threshold)
        # Convert similarity to "confidence" (inverse for backward compatibility)
        confidence = 100 * (1 - similarity) if student_id else 100
        return student_id, confidence


def capture_face_samples(student_id, name, num_samples=10):
    """
    Capture face samples from webcam for enrollment
    """
    cap = cv2.VideoCapture(0)
    yolo = YOLODetector(model_size='n')
    samples = []
    count = 0
    
    print(f"Capturing {num_samples} face samples for {name}...")
    print("Press SPACE to capture, ESC to cancel")
    
    while count < num_samples:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect persons using YOLO
        persons = yolo.detect_persons(frame)
        
        # Draw rectangles around detected persons
        display_frame = frame.copy()
        for (x1, y1, x2, y2, conf) in persons:
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(display_frame, f'{conf:.2f}', (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        cv2.putText(display_frame, f"Captured: {count}/{num_samples}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(display_frame, "Press SPACE to capture", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow('Capture Face Samples', display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == 27:  # ESC
            print("Cancelled")
            break
        
        if key == 32 and len(persons) > 0:  # SPACE
            # Capture the first detected person's face region
            x1, y1, x2, y2, conf = persons[0]
            
            # Extract face region (upper 40%)
            h = y2 - y1
            face_h = int(h * 0.4)
            face_img = frame[y1:y1+face_h, x1:x2]
            
            if face_img.size > 0:
                # Resize to standard size
                face_img = cv2.resize(face_img, (200, 200))
                samples.append(face_img)
                count += 1
                print(f"✓ Captured sample {count}/{num_samples}")
    
    cap.release()
    cv2.destroyAllWindows()
    
    return samples


if __name__ == '__main__':
    # Example usage: Enroll a student with Buffalo_l
    print("=" * 60)
    print("Face Recognition System - YOLOv8n + InsightFace Buffalo_l")
    print("=" * 60)
    
    recognizer = BuffaloFaceRecognizer()
    
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
    else:
        print(f"\n✓ {len(recognizer.known_faces)} students already enrolled")
        print("Run with --enroll to add more students")
