"""
Integrated Inference Pipeline
Combines YOLO detection, Face Recognition, and Behavior Classification
"""

import torch
import torch.nn.functional as F
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from PIL import Image
from torchvision import transforms

from train_behavior_model import BehaviorClassifier
from train_face_model import FaceEmbeddingModel
from behavior_dataset import BehaviorDataset


@dataclass
class StudentDetection:
    """Single student detection result"""
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    student_id: Optional[str] = None
    student_name: Optional[str] = None
    face_confidence: float = 0.0
    behavior: Optional[str] = None
    behavior_confidence: float = 0.0
    embedding: Optional[np.ndarray] = None


class IntegratedStudentAnalyzer:
    """
    Complete student analysis pipeline
    1. YOLO: Detect students
    2. Face Recognition: Identify students
    3. Behavior Classification: Classify behavior
    """
    
    def __init__(
        self,
        yolo_model_path: str = 'yolov8n.pt',
        face_model_path: str = 'models/face/best_face_model.pth',
        behavior_model_path: str = 'models/behavior/best_model.pth',
        face_database_path: str = 'face_embeddings.npy',
        device: str = 'cuda',
        face_threshold: float = 0.5,
        behavior_threshold: float = 0.5
    ):
        self.device = device
        self.face_threshold = face_threshold
        self.behavior_threshold = behavior_threshold
        
        # Load YOLO
        print("Loading YOLO model...")
        try:
            from ultralytics import YOLO
            self.yolo = YOLO(yolo_model_path)
            print(f"  [OK] YOLO loaded from {yolo_model_path}")
        except Exception as e:
            print(f"  [WARNING] Could not load YOLO: {e}")
            print(f"  Install: pip install ultralytics")
            self.yolo = None
        
        # Load Face Recognition Model
        print("Loading Face Recognition model...")
        self.face_model = self._load_face_model(face_model_path)
        
        # Load Face Database
        print("Loading Face Database...")
        self.face_database = self._load_face_database(face_database_path)
        
        # Load Behavior Classification Model
        print("Loading Behavior Classification model...")
        self.behavior_model = self._load_behavior_model(behavior_model_path)
        
        # Image transforms
        self.face_transform = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        self.behavior_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print("\n[OK] All models loaded successfully!\n")
    
    
    def _load_face_model(self, model_path: str) -> Optional[torch.nn.Module]:
        """Load face recognition model"""
        if not Path(model_path).exists():
            print(f"  [WARNING] Face model not found: {model_path}")
            print(f"  Train the model using: python train_face_model.py")
            return None
        
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            embedding_size = checkpoint.get('embedding_size', 512)
            
            model = FaceEmbeddingModel(embedding_size=embedding_size, backbone='resnet50', pretrained=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(self.device)
            model.eval()
            
            print(f"  [OK] Face model loaded (Acc: {checkpoint.get('val_acc', 0):.4f})")
            return model
        except Exception as e:
            print(f"  [ERROR] Failed to load face model: {e}")
            return None
    
    
    def _load_face_database(self, database_path: str) -> Optional[Dict]:
        """Load pre-computed face embeddings database"""
        if not Path(database_path).exists():
            print(f"  [WARNING] Face database not found: {database_path}")
            print(f"  Create database using: python create_face_database.py")
            return None
        
        try:
            data = np.load(database_path, allow_pickle=True).item()
            print(f"  [OK] Face database loaded ({len(data['embeddings'])} students)")
            return data
        except Exception as e:
            print(f"  [ERROR] Failed to load face database: {e}")
            return None
    
    
    def _load_behavior_model(self, model_path: str) -> Optional[torch.nn.Module]:
        """Load behavior classification model"""
        if not Path(model_path).exists():
            print(f"  [WARNING] Behavior model not found: {model_path}")
            print(f"  Train the model using: python train_behavior_model.py")
            return None
        
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            model = BehaviorClassifier(num_classes=7, pretrained=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(self.device)
            model.eval()
            
            print(f"  [OK] Behavior model loaded (Acc: {checkpoint.get('val_acc', 0):.4f})")
            return model
        except Exception as e:
            print(f"  [ERROR] Failed to load behavior model: {e}")
            return None
    
    
    @torch.no_grad()
    def detect_students(self, image: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """
        Detect students using YOLO
        
        Returns:
            List of (x1, y1, x2, y2, confidence)
        """
        if self.yolo is None:
            # Return dummy detection for testing
            h, w = image.shape[:2]
            return [(50, 50, w-50, h-50, 0.9)]
        
        results = self.yolo(image, verbose=False)
        
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Filter for 'person' class (class 0 in COCO)
                if int(box.cls) == 0:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf)
                    detections.append((int(x1), int(y1), int(x2), int(y2), conf))
        
        return detections
    
    
    @torch.no_grad()
    def recognize_face(self, face_crop: np.ndarray) -> Tuple[Optional[str], Optional[str], float]:
        """
        Recognize face using face recognition model
        
        Returns:
            (student_id, student_name, confidence)
        """
        if self.face_model is None or self.face_database is None:
            return None, None, 0.0
        
        # Convert to PIL and apply transform
        face_pil = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
        face_tensor = self.face_transform(face_pil).unsqueeze(0).to(self.device)
        
        # Extract embedding
        embedding = self.face_model(face_tensor)
        embedding = F.normalize(embedding, p=2, dim=1)
        embedding_np = embedding.cpu().numpy()
        
        # Compare with database
        db_embeddings = self.face_database['embeddings']
        db_names = self.face_database['names']
        db_ids = self.face_database['ids']
        
        # Compute cosine similarities
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity(embedding_np, db_embeddings)[0]
        
        # Find best match
        best_idx = np.argmax(similarities)
        best_similarity = similarities[best_idx]
        
        if best_similarity > self.face_threshold:
            return db_ids[best_idx], db_names[best_idx], float(best_similarity)
        else:
            return None, "Unknown", float(best_similarity)
    
    
    @torch.no_grad()
    def classify_behavior(self, person_crop: np.ndarray) -> Tuple[str, float, np.ndarray]:
        """
        Classify student behavior
        
        Returns:
            (behavior_class, confidence, probabilities)
        """
        if self.behavior_model is None:
            return "Unknown", 0.0, np.zeros(7)
        
        # Convert to PIL and apply transform
        crop_pil = Image.fromarray(cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB))
        crop_tensor = self.behavior_transform(crop_pil).unsqueeze(0).to(self.device)
        
        # Predict
        outputs = self.behavior_model(crop_tensor)
        probs = F.softmax(outputs, dim=1)
        
        confidence, pred_idx = torch.max(probs, 1)
        
        behavior_class = BehaviorDataset.CLASSES[pred_idx.item()]
        confidence = float(confidence.item())
        probs_np = probs.cpu().numpy()[0]
        
        return behavior_class, confidence, probs_np
    
    
    def analyze_frame(self, frame: np.ndarray) -> List[StudentDetection]:
        """
        Complete analysis of a single frame
        
        Args:
            frame: BGR image from OpenCV
        
        Returns:
            List of StudentDetection objects
        """
        results = []
        
        # Step 1: Detect students
        detections = self.detect_students(frame)
        
        for bbox in detections:
            x1, y1, x2, y2, det_conf = bbox
            
            # Crop person
            person_crop = frame[y1:y2, x1:x2]
            
            if person_crop.size == 0:
                continue
            
            # Step 2: Recognize face (crop upper part for face)
            h, w = person_crop.shape[:2]
            face_crop = person_crop[0:int(h*0.4), :]  # Top 40% for face
            
            student_id, student_name, face_conf = self.recognize_face(face_crop)
            
            # Step 3: Classify behavior
            behavior, behavior_conf, probs = self.classify_behavior(person_crop)
            
            # Create result
            detection = StudentDetection(
                bbox=(x1, y1, x2, y2),
                confidence=det_conf,
                student_id=student_id,
                student_name=student_name,
                face_confidence=face_conf,
                behavior=behavior,
                behavior_confidence=behavior_conf
            )
            
            results.append(detection)
        
        return results
    
    
    def visualize_results(self, frame: np.ndarray, detections: List[StudentDetection]) -> np.ndarray:
        """Draw bounding boxes and labels on frame"""
        output = frame.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            
            # Color based on behavior
            color_map = {
                'Looking_Forward': (0, 255, 0),      # Green
                'Raising_Hand': (255, 0, 255),       # Magenta
                'Reading': (0, 255, 255),            # Cyan
                'Sleeping': (0, 0, 255),             # Red
                'Standing': (255, 255, 0),           # Yellow
                'Turning_Around': (255, 165, 0),     # Orange
                'Writting': (0, 128, 255)            # Blue-orange
            }
            
            color = color_map.get(det.behavior, (128, 128, 128))
            
            # Draw bbox
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
            
            # Prepare label
            label_lines = []
            if det.student_name:
                label_lines.append(f"{det.student_name} ({det.face_confidence:.2f})")
            if det.behavior:
                label_lines.append(f"{det.behavior} ({det.behavior_confidence:.2f})")
            
            # Draw labels
            y_offset = y1 - 10
            for line in label_lines:
                (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(output, (x1, y_offset - th - 5), (x1 + tw, y_offset), color, -1)
                cv2.putText(output, line, (x1, y_offset - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y_offset -= (th + 10)
        
        return output


# ============= Example Usage =============

def process_video(video_path: str, output_path: str = 'output_video.mp4', sample_rate: int = 30):
    """Process video and save annotated output"""
    
    # Initialize analyzer
    analyzer = IntegratedStudentAnalyzer(
        face_threshold=0.4,
        behavior_threshold=0.3
    )
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps // sample_rate, (width, height))
    
    print(f"\nProcessing video: {video_path}")
    print(f"Total frames: {total_frames}")
    print(f"Sample rate: every {sample_rate} frames")
    print(f"Output: {output_path}\n")
    
    frame_count = 0
    processed_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Sample frames
        if frame_count % sample_rate == 0:
            # Analyze frame
            detections = analyzer.analyze_frame(frame)
            
            # Visualize
            annotated = analyzer.visualize_results(frame, detections)
            
            # Write output
            out.write(annotated)
            
            processed_count += 1
            print(f"Processed frame {frame_count}/{total_frames} ({processed_count} analyzed)", end='\r')
        
        frame_count += 1
    
    cap.release()
    out.release()
    
    print(f"\n\n[DONE] Video saved to {output_path}")


def process_image(image_path: str, output_path: str = 'output_image.jpg'):
    """Process single image"""
    
    # Initialize analyzer
    analyzer = IntegratedStudentAnalyzer(
        face_threshold=0.4,
        behavior_threshold=0.3
    )
    
    # Load image
    frame = cv2.imread(image_path)
    
    # Analyze
    detections = analyzer.analyze_frame(frame)
    
    # Print results
    print(f"\nDetected {len(detections)} students:")
    for i, det in enumerate(detections):
        print(f"\nStudent {i + 1}:")
        print(f"  Name: {det.student_name} (confidence: {det.face_confidence:.4f})")
        print(f"  Behavior: {det.behavior} (confidence: {det.behavior_confidence:.4f})")
        print(f"  BBox: {det.bbox}")
    
    # Visualize
    annotated = analyzer.visualize_results(frame, detections)
    
    # Save
    cv2.imwrite(output_path, annotated)
    print(f"\n[DONE] Image saved to {output_path}")


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python integrated_inference.py <image_path>")
        print("  python integrated_inference.py <video_path>")
        print("\nExample:")
        print("  python integrated_inference.py classroom.jpg")
        print("  python integrated_inference.py classroom_video.mp4")
    else:
        input_path = sys.argv[1]
        
        if input_path.endswith(('.jpg', '.jpeg', '.png')):
            process_image(input_path)
        elif input_path.endswith(('.mp4', '.avi', '.mov')):
            process_video(input_path)
        else:
            print("Unknown file type. Use .jpg, .png, .mp4, .avi, or .mov")

