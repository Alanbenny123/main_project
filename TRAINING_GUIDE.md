# Training Guide

Complete guide for training behavior classification and face recognition models.

## Dataset Overview

### 1. Behavior Dataset (`Behaviors_Features/`)
- **Total Images**: 252,223
- **Classes**: 7 behaviors
  - Looking_Forward (8.43%)
  - Raising_Hand (6.68%)
  - Reading (19.74%)
  - Sleeping (23.95%)
  - Standing (3.14%)
  - Turning_Around (18.61%)
  - Writting (19.45%)
- **Students**: 4 IDs (ID1-ID4)

### 2. Face Recognition Dataset (`NDB/`)
- **Total Students**: 65
- **Total Images**: 759
- **Images per student**: 6-25 (avg 11.7)

## Quick Start

### 1. Analyze Datasets
```bash
python dataset_analyzer.py
```

This will:
- Show statistics for both datasets
- Identify class imbalances
- Generate split recommendations
- Save analysis to `dataset_analysis.json`

### 2. Train Behavior Classification Model

```bash
python train_behavior_model.py
```

**Configuration** (edit in script):
```python
config = {
    'batch_size': 32,
    'num_epochs': 20,
    'learning_rate': 1e-4,
    'model_name': 'swin_tiny_patch4_window7_224',
    'use_class_weights': True,
    'balance_classes': False,  # Set True to balance classes
}
```

**Outputs**:
- `models/behavior/best_model_acc{acc}.pth` - Best model checkpoint
- `models/behavior/training_curves.png` - Training visualization
- `models/behavior/confusion_matrix.png` - Test set confusion matrix
- `models/behavior/training_history.json` - Training metrics

**Training Time**: ~4-8 hours on GPU (depends on dataset size)

### 3. Train Face Recognition Model

```bash
python train_face_model.py
```

**Configuration**:
```python
config = {
    'batch_size': 64,
    'num_epochs': 30,
    'learning_rate': 1e-2,
    'embedding_size': 512,
    'backbone': 'resnet50',
}
```

**Outputs**:
- `models/face/best_face_model_acc{acc}.pth` - Best model
- `models/face/verification_results.json` - Verification metrics

**Training Time**: ~30-60 minutes on GPU

### 4. Create Face Embeddings Database

After training the face model:

```bash
python create_face_database.py
```

**Outputs**:
- `face_embeddings.npy` - Pre-computed embeddings for all students
- `face_embeddings_metadata.json` - Student metadata

### 5. Run Integrated Inference

**Process Image**:
```bash
python integrated_inference.py classroom.jpg
```

**Process Video**:
```bash
python integrated_inference.py classroom_video.mp4
```

## Detailed Training Options

### Behavior Classification

#### Basic Training
```bash
python train_behavior_model.py
```

#### Balanced Classes Training
If you want to balance the dataset (Standing only has 3.14%):

Edit `train_behavior_model.py`:
```python
config = {
    'balance_classes': True,
    'max_samples_per_class': 10000,  # Limit all classes to 10k samples
}
```

#### Custom Data Split
Default split uses ID-based division to prevent data leakage:
- Train: ID1, ID2, ID3
- Val/Test: ID4

To customize, edit `behavior_dataset.py`:
```python
train_loader, val_loader, test_loader = get_behavior_dataloaders(
    train_ids=['ID1', 'ID2'],
    val_ids=['ID3'],
    test_ids=['ID4']
)
```

#### Different Model Backbones
Available Swin Transformer models (edit `train_behavior_model.py`):
```python
# Smaller/faster
model_name = 'swin_tiny_patch4_window7_224'    # 28M params

# Larger/more accurate
model_name = 'swin_small_patch4_window7_224'   # 50M params
model_name = 'swin_base_patch4_window7_224'    # 88M params
```

### Face Recognition

#### Training from Scratch
```python
config = {
    'pretrained': False,  # Don't use ImageNet weights
}
```

#### Different Backbones
```python
# Faster
backbone = 'resnet18'  # 11M params

# Balanced
backbone = 'resnet34'  # 21M params

# Most accurate
backbone = 'resnet50'  # 25M params
```

#### Custom Embedding Size
```python
config = {
    'embedding_size': 256,  # Smaller/faster
    # or
    'embedding_size': 1024,  # Larger/more accurate
}
```

## Model Architectures

### Behavior Classification
```
Input Image (224x224)
    ↓
Swin Transformer Backbone
    ↓
Classification Head
    ↓
7 Classes (behaviors)
```

### Face Recognition
```
Input Face (112x112)
    ↓
ResNet Backbone
    ↓
Embedding Layer (512-d)
    ↓
ArcFace Loss (training)
    ↓
Cosine Similarity (inference)
```

## Performance Tips

### GPU Optimization
```python
# Increase batch size for faster training
batch_size = 64  # or 128 if you have GPU memory

# Increase num_workers for faster data loading
num_workers = 8  # adjust based on your CPU cores
```

### CPU Training
If you don't have a GPU:

```python
config = {
    'device': 'cpu',
    'batch_size': 16,  # Smaller batch
    'num_workers': 2,
}
```

**Note**: Training will be much slower (10-20x)

## Evaluation Metrics

### Behavior Classification
- **Accuracy**: Overall classification accuracy
- **Per-class Precision/Recall/F1**: Detailed per-behavior metrics
- **Confusion Matrix**: Visualizes misclassifications

### Face Recognition
- **Identification Accuracy**: Can we identify the correct student?
- **Verification Accuracy**: Can we verify if two faces are the same?
- **Precision/Recall**: True positive vs false positive rate

## Common Issues

### 1. Out of Memory
```
RuntimeError: CUDA out of memory
```

**Solution**: Reduce batch size
```python
batch_size = 16  # or 8
```

### 2. Class Imbalance
If some behaviors perform poorly:

**Solution**: Use class weights
```python
use_class_weights = True  # Already default
```

Or balance dataset:
```python
balance_classes = True
max_samples_per_class = 10000
```

### 3. Overfitting
If validation accuracy plateaus but train keeps improving:

**Solution**: 
- Add more data augmentation
- Increase weight decay
- Reduce model size
- Early stopping (already implemented)

### 4. Slow Training
**Solutions**:
- Reduce `max_frames` in config
- Use smaller model (swin_tiny instead of swin_base)
- Increase num_workers
- Use GPU if available

## Directory Structure After Training

```
mma/
├── Behaviors_Features/        # Behavior dataset
├── NDB/                        # Face dataset
├── models/
│   ├── behavior/
│   │   ├── best_model.pth
│   │   ├── training_curves.png
│   │   └── confusion_matrix.png
│   └── face/
│       ├── best_face_model.pth
│       └── verification_results.json
├── face_embeddings.npy         # Pre-computed face embeddings
├── dataset_analysis.json       # Dataset statistics
├── behavior_dataset.py         # Behavior data loader
├── face_dataset.py             # Face data loader
├── train_behavior_model.py     # Behavior training script
├── train_face_model.py         # Face training script
├── create_face_database.py     # Create embeddings database
└── integrated_inference.py     # Full inference pipeline
```

## Next Steps

After training:

1. **Integrate with Video Preprocessing**:
   ```python
   from video_preprocessing import VideoPreprocessor
   from integrated_inference import IntegratedStudentAnalyzer
   
   preprocessor = VideoPreprocessor()
   analyzer = IntegratedStudentAnalyzer()
   
   # Process classroom video
   for frame_num, frame in preprocessor.extract_frames('classroom.mp4'):
       detections = analyzer.analyze_frame(frame)
       # ... store results
   ```

2. **Connect to Database**:
   - Use `database_hybrid.py` to store analysis results
   - Track student behaviors over time
   - Generate reports

3. **Deploy to Flask API**:
   - Add endpoints to `app.py`
   - Enable real-time analysis
   - Create dashboard

## Support

For issues or questions:
1. Check this guide
2. Review code comments in training scripts
3. Examine dataset with `dataset_analyzer.py`

