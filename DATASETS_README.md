# Dataset Documentation

## Overview

This project uses two main datasets:

1. **Behaviors_Features/** - Student behavior classification (252K images, 7 classes)
2. **NDB/** - Student face recognition (759 images, 65 students)

## Dataset 1: Behavior Classification

### Location
```
Behaviors_Features/
├── Looking_Forward/
│   ├── ID1/
│   ├── ID2/
│   ├── ID3/
│   └── ID4/
├── Raising_Hand/
├── Reading/
├── Sleeping/
├── Standing/
├── Turning_Around/
└── Writting/
```

### Statistics

| Behavior | Total Images | Percentage | ID1 | ID2 | ID3 | ID4 |
|----------|-------------|------------|-----|-----|-----|-----|
| Looking_Forward | 21,269 | 8.43% | 5,382 | 5,422 | 4,285 | 6,180 |
| Raising_Hand | 16,845 | 6.68% | 2,999 | 3,523 | 4,948 | 5,375 |
| Reading | 49,799 | 19.74% | 18,564 | 17,614 | 3,050 | 10,571 |
| Sleeping | 60,397 | 23.95% | 16,339 | 14,988 | 15,877 | 13,193 |
| Standing | 7,931 | 3.14% | 2,353 | 1,665 | 1,902 | 2,011 |
| Turning_Around | 46,936 | 18.61% | 12,574 | 11,894 | 11,977 | 10,491 |
| Writting | 49,046 | 19.45% | 12,295 | 11,755 | 12,014 | 12,982 |
| **TOTAL** | **252,223** | **100%** | **70,506** | **66,861** | **54,053** | **60,803** |

### Image Format
- **Type**: PNG
- **Organization**: Each behavior → Student ID → Multiple video sequences → Frames
- **Example**: `Behaviors_Features/Reading/ID1/Forward1_id1_Act1_rgb/*.png`

### Data Split Recommendation
To prevent data leakage (same person in train/test):
- **Train**: ID1, ID2, ID3 (75%)
- **Validation/Test**: ID4 (25%)

### Class Balance Notes
- **Imbalanced**: Standing (3.14%) is underrepresented
- **Dominant**: Sleeping (23.95%), Reading (19.74%), Writting (19.45%)
- **Solution**: Use class weights in loss function or balance during training

## Dataset 2: Face Recognition

### Location
```
NDB/
├── ASI22CA001_Aaliya_M_Ismail/
│   ├── IMG20250717103528.jpg
│   ├── IMG20250717103530.jpg
│   └── ...
├── ASI22CA002_AAQUIB_HANAN/
├── ASI22CA003_Abdul_Razak_K_K/
└── ...
```

### Statistics
- **Total Students**: 65
- **Total Images**: 759
- **Images per Student**: 6-25 (average: 11.7)

### Image Format
- **Type**: JPG
- **Naming**: `{StudentID}_{FirstName}_{LastName}/IMG*.jpg`
- **Example**: `ASI22CA009_Alan_Benny/IMG20250717103528.jpg`

### Student ID Format
- **Pattern**: `ASI22CA{number}`
- **Example**: ASI22CA009

### Data Split Recommendation
Per-student split (to ensure all students in train/val/test):
- **Train**: 70% of each student's images
- **Validation**: 15% of each student's images
- **Test**: 15% of each student's images

## Usage

### Load Behavior Dataset
```python
from behavior_dataset import BehaviorDataset, get_behavior_dataloaders

# Single dataset
dataset = BehaviorDataset(
    student_ids=['ID1', 'ID2'],
    augment=True
)

# Or get train/val/test loaders
train_loader, val_loader, test_loader = get_behavior_dataloaders(
    batch_size=32,
    train_ids=['ID1', 'ID2', 'ID3'],
    val_ids=['ID4'],
    test_ids=['ID4']
)
```

### Load Face Dataset
```python
from face_dataset import FaceDataset, get_face_dataloaders

# Single dataset
dataset = FaceDataset(
    split='train',
    augment=True
)

# Or get train/val/test loaders
train_loader, val_loader, test_loader, num_classes = get_face_dataloaders(
    batch_size=64,
    split_ratio=(0.70, 0.15, 0.15)
)
```

### Analyze Datasets
```python
from dataset_analyzer import DatasetAnalyzer

analyzer = DatasetAnalyzer()
behavior_stats = analyzer.analyze_behaviors()
face_stats = analyzer.analyze_faces()
analyzer.save_analysis('analysis.json')
```

Or run from command line:
```bash
python dataset_analyzer.py
```

## Data Quality

### Behavior Dataset
- **✓ Good**: Large dataset (252K images)
- **✓ Good**: Multiple students (4 IDs) prevents overfitting
- **⚠ Warning**: Class imbalance (Standing only 3.14%)
- **✓ Solution**: Use weighted loss or balanced sampling

### Face Dataset
- **✓ Good**: All students have >= 6 images
- **✓ Good**: Reasonable images per student (avg 11.7)
- **⚠ Limitation**: Small number of images per student
- **✓ Solution**: Use data augmentation during training

## Data Augmentation

### Behavior Classification
Applied during training:
- Random crop (224x224 from 256x256)
- Random horizontal flip (50%)
- Random rotation (±10°)
- Color jitter (brightness, contrast, saturation)
- Normalization (ImageNet stats)

### Face Recognition
Applied during training:
- Random crop (112x112 from 128x128)
- Random horizontal flip (50%)
- Color jitter (brightness, contrast)
- Random rotation (±15°)
- Normalization ([-1, 1] range)

## File Organization

Each dataset has:
- **Data loader**: `behavior_dataset.py`, `face_dataset.py`
- **Training script**: `train_behavior_model.py`, `train_face_model.py`
- **Analysis**: `dataset_analyzer.py`

## References

### Behavior Classes
1. **Looking_Forward**: Student facing forward/teacher
2. **Raising_Hand**: Student raising hand to answer
3. **Reading**: Student reading books/materials
4. **Sleeping**: Student sleeping on desk
5. **Standing**: Student standing up
6. **Turning_Around**: Student turned away from front
7. **Writting**: Student writing in notebook

### Model Targets
- **Behavior**: 7-class classification (multi-class)
- **Face**: 65-class identification (during training) → Embedding-based verification (during inference)

## Next Steps

1. **Analyze**: `python dataset_analyzer.py`
2. **Train Behavior Model**: `python train_behavior_model.py`
3. **Train Face Model**: `python train_face_model.py`
4. **Create Face Database**: `python create_face_database.py`
5. **Run Inference**: `python integrated_inference.py <image>`

See **TRAINING_GUIDE.md** for detailed instructions.

