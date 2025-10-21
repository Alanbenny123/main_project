# Algorithm Setup Guide

Complete guide for setting up YOLOv8n and InsightFace Buffalo_l in the Student Behavior Analysis System.

---

## Overview

This system uses **three core algorithms**:

1. **Swin Transformer** → Behavior classification (Raising Hand, Reading, Sleeping, Writing)
2. **YOLOv8n** → Person/object detection in classroom videos
3. **InsightFace Buffalo_l** → Facial recognition for student identification

---

## 1. YOLOv8n Setup (Person Detection)

### What is YOLOv8n?

**YOLO** (You Only Look Once) is a state-of-the-art real-time object detection algorithm.
- **v8n** = Version 8, Nano variant (fastest, smallest)
- Detects 80 object classes including **persons**
- Single-pass detection (no region proposals needed)

### Installation

```bash
pip install ultralytics>=8.0.0
```

### Auto-Download

YOLOv8n model auto-downloads on first run (~6MB):
```python
from ultralytics import YOLO
model = YOLO('yolov8n.pt')  # Downloads if not present
```

**Model will be saved to**: `~/.cache/ultralytics/`

### Manual Download (Optional)

```bash
# Download YOLOv8n weights
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
```

### How It Works

```python
# Detect persons in frame
results = model(frame, conf=0.5, classes=[0])  # class 0 = person
boxes = results[0].boxes

for box in boxes:
    x1, y1, x2, y2 = box.xyxy[0]  # Bounding box coordinates
    confidence = box.conf[0]       # Detection confidence (0-1)
```

### Parameters

- `conf=0.5` → Minimum confidence threshold (50%)
- `classes=[0]` → Filter for person class only
- `det_size=(640, 640)` → Detection resolution

### GPU Acceleration (Optional)

```python
# Use CUDA if available
model = YOLO('yolov8n.pt')
results = model(frame, device='cuda')  # or device=0 for GPU 0
```

---

## 2. InsightFace Buffalo_l Setup (Face Recognition)

### What is InsightFace?

**InsightFace** is a state-of-the-art face recognition library.
- **Buffalo_l** = Large variant with 99.8% accuracy
- Uses **ArcFace loss** for training
- Outputs **512-dimensional embeddings**

### Installation

```bash
pip install insightface>=0.7.3
pip install onnxruntime>=1.15.0  # CPU runtime

# For GPU acceleration (optional)
pip install onnxruntime-gpu>=1.15.0
```

### Model Download

Buffalo_l model auto-downloads on first use (~350MB):

```python
from insightface.app import FaceAnalysis

app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0, det_size=(640, 640))
```

**Models saved to**: `~/.insightface/models/buffalo_l/`

### Manual Download (if needed)

```bash
# Create directory
mkdir -p ~/.insightface/models/buffalo_l/

# Download from official repo
# Visit: https://github.com/deepinsight/insightface/tree/master/python-package
```

### How It Works

#### Step 1: Extract Face Embeddings

```python
# Get face embedding (512D vector)
faces = app.get(image)
embedding = faces[0].embedding  # numpy array, shape: (512,)
```

#### Step 2: Enroll Students

```python
# Collect 5-10 face images per student
embeddings = [app.get(img)[0].embedding for img in face_images]

# Average embeddings for robustness
avg_embedding = np.mean(embeddings, axis=0)

# Store with student ID
student_db[student_id] = avg_embedding
```

#### Step 3: Recognize Faces

```python
# Query face
query_embedding = app.get(query_image)[0].embedding

# Compare with all enrolled students using cosine similarity
for student_id, stored_embedding in student_db.items():
    similarity = np.dot(query_embedding, stored_embedding) / (
        np.linalg.norm(query_embedding) * np.linalg.norm(stored_embedding)
    )
    
    if similarity >= 0.4:  # Threshold
        return student_id, similarity
```

### Cosine Similarity

**Formula**:
```
similarity = (A · B) / (||A|| × ||B||)
```

**Range**: -1 to 1 (higher = more similar)
- `0.4` → Different person (threshold)
- `0.6` → Likely same person
- `0.8+` → Very confident match

### GPU Acceleration

```python
# Use CUDA
app = FaceAnalysis(
    name='buffalo_l',
    providers=['CUDAExecutionProvider']  # GPU
)

# Or CPU only
app = FaceAnalysis(
    name='buffalo_l',
    providers=['CPUExecutionProvider']  # CPU
)
```

---

## 3. Swin Transformer Setup (Behavior Classification)

See [MODEL_SETUP.md](MODEL_SETUP.md) for detailed instructions on training the Swin Transformer.

**Quick Summary**:
- Download dataset from Kaggle
- Train using provided Jupyter notebook
- Model saved to `./saved_model/student_behavior_model.h5`

---

## Integration Pipeline

### Full Classroom Analysis Flow

```
1. Load video frame
   ↓
2. YOLOv8n detects all persons → Bounding boxes
   ↓
3. For each person:
   a. Extract face region (upper 40% of bbox)
   b. InsightFace extracts 512D embedding
   c. Compare with enrolled students → student_id
   d. Swin Transformer predicts behavior
   ↓
4. Aggregate results per student
   ↓
5. Generate individual reports
```

### Code Example

```python
from ultralytics import YOLO
from insightface.app import FaceAnalysis
import tensorflow as tf

# Load models
yolo = YOLO('yolov8n.pt')
face_app = FaceAnalysis(name='buffalo_l')
face_app.prepare(ctx_id=0, det_size=(640, 640))
behavior_model = tf.keras.models.load_model('saved_model/student_behavior_model.h5')

# Analyze frame
def analyze_classroom(frame):
    results = []
    
    # 1. Detect persons
    persons = yolo(frame, conf=0.5, classes=[0])
    
    for box in persons[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        
        # 2. Extract person region
        person_img = frame[int(y1):int(y2), int(x1):int(x2)]
        
        # 3. Get face embedding
        faces = face_app.get(person_img)
        if len(faces) > 0:
            embedding = faces[0].embedding
            student_id = match_student(embedding)  # Your matching function
            
            # 4. Predict behavior
            preprocessed = preprocess(person_img)
            behavior = behavior_model.predict(preprocessed)
            
            results.append({
                'student_id': student_id,
                'behavior': behavior,
                'bbox': (x1, y1, x2, y2)
            })
    
    return results
```

---

## Performance Benchmarks

### YOLOv8n
- **CPU (Intel i7)**: ~20-30 FPS
- **GPU (RTX 3060)**: ~100-150 FPS
- **Model Size**: 6 MB
- **Inference Time**: 30-50ms (CPU)

### InsightFace Buffalo_l
- **CPU**: ~50ms per face
- **GPU**: ~10ms per face
- **Model Size**: 350 MB
- **Accuracy**: 99.8% (LFW benchmark)

### Swin Transformer
- **CPU**: ~100-200ms per frame
- **GPU**: ~20-30ms per frame
- **Model Size**: ~150 MB (depends on training)

---

## Troubleshooting

### YOLOv8n Issues

**Problem**: `ModuleNotFoundError: No module named 'ultralytics'`
```bash
pip install ultralytics
```

**Problem**: Model fails to download
```bash
# Check internet connection
# Or manually download from: https://github.com/ultralytics/assets/releases
```

**Problem**: CUDA not available
```python
# Force CPU mode
results = model(frame, device='cpu')
```

### InsightFace Issues

**Problem**: `ONNXRuntimeError`
```bash
pip install --upgrade onnxruntime
```

**Problem**: Model not found
```bash
# Clear cache and re-download
rm -rf ~/.insightface/
python -c "from insightface.app import FaceAnalysis; app = FaceAnalysis(name='buffalo_l'); app.prepare(ctx_id=0)"
```

**Problem**: No faces detected
- Check image quality (resolution, lighting)
- Increase detection size: `det_size=(1280, 1280)`
- Verify face is visible and frontal

### Memory Issues

**Problem**: Out of memory during video processing
```python
# Reduce detection size
det_size=(320, 320)  # Lower resolution

# Process fewer frames
sample_rate=60  # Every 60 frames instead of 30
```

---

## Optimization Tips

### 1. Batch Processing (for videos)

```python
# Process multiple frames at once
frames = [frame1, frame2, frame3]
results = model(frames)  # Faster than individual calls
```

### 2. Multi-threading

```python
from concurrent.futures import ThreadPoolExecutor

def process_frame(frame):
    return yolo(frame)

with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(process_frame, frames))
```

### 3. Resolution Tuning

```python
# Lower resolution = faster, less accurate
yolo(..., imgsz=320)  # vs default 640

# Higher resolution = slower, more accurate
yolo(..., imgsz=1280)
```

### 4. Model Quantization (Advanced)

```bash
# Convert to TensorRT for 2-5x speedup on NVIDIA GPUs
yolo export model=yolov8n.pt format=engine device=0
```

---

## Testing the Setup

### Test YOLOv8n

```bash
cd tests/
python test_yolo.py
```

```python
# test_yolo.py
from ultralytics import YOLO
import cv2

model = YOLO('yolov8n.pt')
frame = cv2.imread('test_classroom.jpg')
results = model(frame, conf=0.5, classes=[0])

print(f"Detected {len(results[0].boxes)} persons")
results[0].show()  # Display with bounding boxes
```

### Test InsightFace

```bash
python test_insightface.py
```

```python
# test_insightface.py
from insightface.app import FaceAnalysis
import cv2

app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0)

img = cv2.imread('test_face.jpg')
faces = app.get(img)

print(f"Detected {len(faces)} faces")
for face in faces:
    print(f"Embedding shape: {face.embedding.shape}")
    print(f"Bbox: {face.bbox}")
```

### Test Full Pipeline

```bash
python face_recognition_system.py
```

This will:
1. Initialize YOLOv8n
2. Initialize InsightFace Buffalo_l
3. Prompt to enroll students via webcam
4. Test recognition on new captures

---

## Next Steps

1. ✅ Install dependencies: `pip install -r requirements.txt`
2. ✅ Test YOLOv8n: Run person detection test
3. ✅ Test InsightFace: Run face recognition test
4. ✅ Enroll students: Use webcam or upload images
5. ✅ Train behavior model: See [MODEL_SETUP.md](MODEL_SETUP.md)
6. ✅ Run full system: `python app.py` + `npm run dev`

---

## References

- **YOLOv8**: https://docs.ultralytics.com/
- **InsightFace**: https://github.com/deepinsight/insightface
- **Buffalo_l Paper**: https://arxiv.org/abs/2207.13084
- **ArcFace**: https://arxiv.org/abs/1801.07698
- **Swin Transformer**: https://arxiv.org/abs/2103.14030

---

## License

- YOLOv8: AGPL-3.0
- InsightFace: MIT
- Swin Transformer: Apache 2.0

