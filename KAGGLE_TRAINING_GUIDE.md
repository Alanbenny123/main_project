# 🚀 Kaggle Training Guide

Complete guide for training your student behavior classification model on Kaggle's free GPU.

---

## 📋 Table of Contents
1. [Why Kaggle?](#why-kaggle)
2. [Prerequisites](#prerequisites)
3. [Step-by-Step Setup](#step-by-step-setup)
4. [Running the Training](#running-the-training)
5. [Monitoring Progress](#monitoring-progress)
6. [Downloading Results](#downloading-results)
7. [Using the Model Locally](#using-the-model-locally)
8. [Troubleshooting](#troubleshooting)

---

## 🎯 Why Kaggle?

**Kaggle Advantages:**
- ✅ Free Tesla P100 GPU (16GB VRAM)
- ✅ 30 hours/week GPU time
- ✅ Fast training (~1.5-2 hours for 25 epochs)
- ✅ No local GPU wear and tear
- ✅ Can close laptop and check back later

**Your GTX 1650 vs Kaggle P100:**
```
Task                    GTX 1650 (4GB)    Kaggle P100 (16GB)
─────────────────────────────────────────────────────────────
Batch Size              16-24             64+
Training Time (25 ep)   ~8-10 hours       ~1.5-2 hours
Speed Multiplier        1x                4-5x faster
Risk of Overheat        High              None
```

---

## ✅ Prerequisites

### 1. Kaggle Account
- Create free account at [kaggle.com](https://www.kaggle.com/)
- Verify your phone number (required for GPU access)

### 2. Dataset Preparation
You have **two options** for uploading your dataset:

#### **Option A: Upload as Kaggle Dataset (Recommended)**
- Easier to reuse across notebooks
- Faster loading

#### **Option B: Upload as ZIP file**
- Simpler for one-time use
- Need to extract in notebook

---

## 📦 Step-by-Step Setup

### Step 1: Prepare Your Dataset

**Create a ZIP file of your dataset:**

```bash
# On your local machine
cd C:\Users\alanb\Downloads\mma
zip -r Behaviors_Features.zip Behaviors_Features/
```

Or on Windows (PowerShell):
```powershell
Compress-Archive -Path "Behaviors_Features" -DestinationPath "Behaviors_Features.zip"
```

**Your dataset structure should be:**
```
Behaviors_Features/
├── Looking_Forward/
│   ├── ID1/ (PNG files)
│   ├── ID2/ (PNG files)
│   ├── ID3/ (PNG files)
│   └── ID4/ (PNG files)
├── Raising_Hand/
│   ├── ID1/
│   ├── ID2/
│   ├── ID3/
│   └── ID4/
├── Reading/
├── Sleeping/
├── Standing/
├── Turning_Around/
└── Writting/
```

### Step 2: Upload Dataset to Kaggle

#### **Option A: Create Kaggle Dataset**

1. Go to https://www.kaggle.com/datasets
2. Click **"New Dataset"**
3. Upload `Behaviors_Features.zip`
4. Title: `student-behavior-dataset` (or your choice)
5. Make it **Private**
6. Click **"Create"**
7. **Note the dataset path** (will be like `/kaggle/input/student-behavior-dataset/`)

#### **Option B: Upload to Notebook Directly**

1. Create new notebook first (see Step 3)
2. Click **"Add Data"** → **"Upload"**
3. Upload `Behaviors_Features.zip`

### Step 3: Create Kaggle Notebook

1. Go to https://www.kaggle.com/code
2. Click **"New Notebook"**
3. Click **"File"** → **"Upload Notebook"**
4. Upload `kaggle_training_notebook.ipynb` from your project
5. **Configure notebook settings:**
   - Click **Settings** (gear icon) on the right
   - **Accelerator:** GPU P100 ✅
   - **Internet:** ON ✅ (needed for downloading pretrained model)
   - **Language:** Python
   - Click **Save**

### Step 4: Link Dataset to Notebook

**If you used Option A (Kaggle Dataset):**
1. Click **"Add Data"** in the right sidebar
2. Search for your dataset name
3. Click **"Add"**
4. Note the path shown (like `/kaggle/input/student-behavior-dataset/`)

**If you used Option B (Direct Upload):**
- Dataset will be at `/kaggle/input/[filename]/`

---

## 🏃 Running the Training

### 1. Update Dataset Path

Find this cell in the notebook:
```python
# ===== ADJUST THIS PATH =====
DATASET_PATH = "/kaggle/input/your-dataset-name/Behaviors_Features"
```

**Change to your actual path:**

For Kaggle Dataset:
```python
DATASET_PATH = "/kaggle/input/student-behavior-dataset/Behaviors_Features"
```

For uploaded ZIP (extract first):
```python
!unzip -q /kaggle/input/behaviors-features-zip/Behaviors_Features.zip -d /kaggle/working/
DATASET_PATH = "/kaggle/working/Behaviors_Features"
```

### 2. Adjust Configuration (Optional)

You can modify training parameters:

```python
CONFIG = {
    'batch_size': 64,           # Increase to 96 if you want faster training
    'num_epochs': 25,           # Reduce to 15 for quick test
    'learning_rate': 1e-4,      # Keep default
    'weight_decay': 1e-4,       # Keep default
    
    'model_name': 'swin_tiny_patch4_window7_224',  # Keep default
    'pretrained': True,         # Always True
    'use_class_weights': True,  # True for imbalanced data
}
```

**Recommended settings:**
- **Quick test (1 hour):** `num_epochs: 15`, `batch_size: 64`
- **Full training (2 hours):** `num_epochs: 25`, `batch_size: 64`
- **Max performance (2.5 hours):** `num_epochs: 30`, `batch_size: 96`

### 3. Run All Cells

**Option A: Run all at once**
1. Click **"Run All"** at the top
2. Go grab coffee ☕

**Option B: Run cell by cell**
1. Click on first cell
2. Press **Shift + Enter** to run and move to next
3. Repeat for each cell

---

## 📊 Monitoring Progress

### What You'll See

**1. Dataset Loading (~2-3 minutes)**
```
✅ Dataset found at: /kaggle/input/student-behavior-dataset/Behaviors_Features
Loaded 89,123 samples from 3 students

📊 Class Distribution:
Looking_Forward      18,234 (20.45%) ██████████
Raising_Hand          8,923 (10.01%) █████
...
```

**2. Model Creation (~30 seconds)**
```
✅ Model created:
   Architecture: swin_tiny_patch4_window7_224
   Total params: 28,288,327
   Trainable:    28,288,327
```

**3. Training Progress**
```
Epoch 1/25
Training: 100%|████████| 1200/1200 [03:24<00:00, 5.88it/s, loss=1.234, acc=0.567]
Validation: 100%|███████| 400/400 [00:42<00:00, 9.52it/s]

📊 Results:
   Train Loss: 1.2340 | Train Acc: 0.5678
   Val Loss:   1.0234 | Val Acc:   0.6234
   ✅ Saved new best model (acc: 0.6234)
```

**4. Final Results**
```
🎉 TRAINING COMPLETE!
Total time: 98.3 minutes
Best validation accuracy: 0.8734
```

### Estimated Timeline

For **100,000 images, 25 epochs, batch size 64:**

| Phase                | Time      | What's Happening                 |
|----------------------|-----------|----------------------------------|
| Setup & Install      | 2 min     | Installing packages              |
| Dataset Loading      | 3 min     | Loading image paths              |
| Model Download       | 1 min     | Downloading pretrained weights   |
| Epoch 1-5            | 25 min    | Initial training                 |
| Epoch 6-15           | 50 min    | Main training                    |
| Epoch 16-25          | 50 min    | Fine-tuning                      |
| Evaluation & Plots   | 5 min     | Testing and visualization        |
| **TOTAL**            | **~2 hrs**| Complete pipeline                |

---

## 💾 Downloading Results

### What Gets Saved

All files are in `/kaggle/working/models/`:

| File                       | Size     | Description                           |
|----------------------------|----------|---------------------------------------|
| `best_behavior_model.pth`  | ~110 MB  | **Main model checkpoint** ⭐          |
| `training_history.json`    | ~5 KB    | Loss/accuracy per epoch               |
| `test_results.json`        | ~10 KB   | Detailed evaluation metrics           |
| `training_curves.png`      | ~200 KB  | Training visualization                |
| `confusion_matrix.png`     | ~300 KB  | Confusion matrix heatmap              |
| `checkpoint_epoch5.pth`    | ~110 MB  | Backup checkpoints                    |
| `checkpoint_epoch10.pth`   | ~110 MB  | (optional)                            |

### How to Download

**Method 1: From Notebook Output**
1. Click **"Output"** tab on the right sidebar
2. Click download icon next to each file
3. Or click **"Download all"** to get everything as ZIP

**Method 2: From Notebook Versions**
1. Click **"Save Version"** at top right
2. Choose **"Save & Run All"**
3. Once complete, go to **"Output"** tab
4. Download files

**Method 3: Programmatic Download**
Add this cell at the end:
```python
# Create a single archive of all outputs
!zip -r /kaggle/working/training_outputs.zip /kaggle/working/models/

print("✅ Created training_outputs.zip - download from Output tab!")
```

---

## 🏠 Using the Model Locally

### 1. Download the Model

Download `best_behavior_model.pth` to your local project:
```
C:\Users\alanb\Downloads\mma\models\behavior\best_behavior_model.pth
```

### 2. Load the Model in Your Code

```python
import torch
import timm
from train_behavior_model import BehaviorClassifier

# Create model architecture
model = BehaviorClassifier(
    num_classes=7,
    pretrained=False,  # Don't download again
    model_name='swin_tiny_patch4_window7_224'
)

# Load trained weights
checkpoint = torch.load('models/behavior/best_behavior_model.pth', 
                       map_location='cpu')  # or 'cuda'
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"✅ Loaded model trained to {checkpoint['val_acc']:.4f} accuracy")
print(f"   Trained for {checkpoint['epoch']} epochs")
```

### 3. Run Inference

```python
from PIL import Image
from torchvision import transforms

# Load and preprocess image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
])

image = Image.open('test_image.png').convert('RGB')
image_tensor = transform(image).unsqueeze(0)

# Predict
with torch.no_grad():
    output = model(image_tensor)
    probs = torch.softmax(output, dim=1)
    pred_class = torch.argmax(probs, dim=1).item()

classes = ['Looking_Forward', 'Raising_Hand', 'Reading', 'Sleeping',
          'Standing', 'Turning_Around', 'Writting']

print(f"Prediction: {classes[pred_class]} ({probs[0][pred_class]:.2%})")
```

### 4. Integrate with Your App

The model file works with your existing `integrated_inference.py`:

```python
# In integrated_inference.py, just update the path:
BEHAVIOR_MODEL_PATH = 'models/behavior/best_behavior_model.pth'

# Everything else stays the same!
```

---

## 🔧 Troubleshooting

### Common Issues

#### **1. Dataset Not Found**
```
❌ Dataset NOT found at: /kaggle/input/...
```

**Solution:**
- Check the dataset is added to notebook (click "Add Data")
- Verify the path in the dataset listing
- Update `DATASET_PATH` to match exactly

#### **2. Out of Memory (OOM)**
```
RuntimeError: CUDA out of memory
```

**Solution:**
```python
CONFIG = {
    'batch_size': 32,  # Reduce from 64
    # ... rest of config
}
```

#### **3. No GPU Available**
```
GPU: None
CUDA: False
```

**Solution:**
- Settings → Accelerator → GPU P100
- Click Save
- Restart notebook

#### **4. Internet Connection Required**
```
Error downloading pretrained model
```

**Solution:**
- Settings → Internet → ON
- Click Save
- Rerun the cell

#### **5. Training Takes Too Long**
```
Training stuck at epoch 1...
```

**Check:**
- Is `num_workers: 2`? (not 0 or 8)
- Is GPU enabled? (should see CUDA: True)
- Dataset loading correctly? (should be ~100K samples)

**Quick test with small dataset:**
```python
train_dataset = BehaviorDataset(
    root_dir=DATASET_PATH,
    student_ids=['ID1'],  # Just one student
    augment=True,
    max_samples_per_class=1000  # Limit samples
)
```

#### **6. Notebook Times Out**
```
Notebook exceeded maximum runtime
```

**Cause:** Kaggle has 9-hour session limit

**Solution:**
- Save version before it times out
- Reduce `num_epochs` to fit in time
- Enable auto-save: Click "Save Version" → "Quick Save" every 5 epochs

---

## ⏱️ GPU Time Management

**Kaggle Free Tier Limits:**
- 30 hours/week GPU time
- 9 hours per session maximum

**Your Training Usage:**
- 25 epochs: ~2 hours
- You can run **~15 full trainings per week**

**Tips to Save Time:**
1. **Test with small dataset first:**
   ```python
   max_samples_per_class=1000  # 5 minutes to verify everything works
   ```

2. **Start with fewer epochs:**
   ```python
   'num_epochs': 15  # ~1.2 hours instead of 2 hours
   ```

3. **Use checkpoints:**
   - If interrupted, load from last checkpoint
   ```python
   checkpoint = torch.load('/kaggle/working/models/checkpoint_epoch15.pth')
   model.load_state_dict(checkpoint['model_state_dict'])
   optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
   start_epoch = checkpoint['epoch']
   ```

---

## 📈 Expected Results

### Typical Performance

**After 25 epochs with balanced dataset:**

| Metric               | Expected Range |
|----------------------|----------------|
| Training Accuracy    | 85-95%         |
| Validation Accuracy  | 80-90%         |
| Test Accuracy        | 78-88%         |
| Training Time        | 1.5-2.5 hours  |

**Per-Class Performance (F1-Score):**
- Looking_Forward: 0.85-0.92
- Raising_Hand: 0.75-0.88
- Reading: 0.82-0.90
- Sleeping: 0.88-0.95 (easiest to detect)
- Standing: 0.78-0.86
- Turning_Around: 0.72-0.85
- Writting: 0.80-0.88

### What to Do if Results are Poor

**Validation accuracy < 70%:**
1. Increase epochs to 30
2. Try balanced dataset:
   ```python
   max_samples_per_class=8000
   ```
3. Increase batch size to 96

**Overfitting (train acc >> val acc):**
1. Use data augmentation (already enabled)
2. Increase weight decay:
   ```python
   'weight_decay': 5e-4  # from 1e-4
   ```
3. Add dropout (requires model modification)

**Underfitting (both accuracies low):**
1. Increase learning rate:
   ```python
   'learning_rate': 2e-4  # from 1e-4
   ```
2. Train longer (30-40 epochs)
3. Try larger model:
   ```python
   'model_name': 'swin_small_patch4_window7_224'  # 50M params
   ```

---

## 🎓 Next Steps

After downloading your trained model:

1. **Test locally:**
   ```bash
   python integrated_inference.py --test_image test.png
   ```

2. **Integrate with frontend:**
   - Model already works with your Flask backend
   - Just update path in `app.py`

3. **Deploy:**
   - Model works on CPU (just slower)
   - For production, use your GTX 1650 for real-time inference

4. **Improve further:**
   - Collect more data for weak classes
   - Try ensemble with multiple models
   - Fine-tune on your specific classroom setup

---

## 📞 Support

**If you get stuck:**

1. Check the notebook output logs
2. Review this guide's troubleshooting section
3. Kaggle Discussion Forums: https://www.kaggle.com/discussions
4. Verify dataset structure matches exactly

**Quick Debugging Cell:**
```python
# Add this cell to debug issues
import sys
print("Python:", sys.version)
print("PyTorch:", torch.__version__)
print("CUDA:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")
print("Dataset path exists:", Path(DATASET_PATH).exists())
if Path(DATASET_PATH).exists():
    print("Classes found:", [d.name for d in Path(DATASET_PATH).iterdir() if d.is_dir()])
```

---

## ✅ Checklist

Before starting training:

- [ ] Kaggle account created and phone verified
- [ ] Dataset zipped and ready
- [ ] Dataset uploaded to Kaggle
- [ ] Notebook uploaded to Kaggle
- [ ] GPU enabled (Settings → Accelerator → GPU P100)
- [ ] Internet enabled (Settings → Internet → ON)
- [ ] Dataset path updated in notebook
- [ ] Configuration reviewed
- [ ] Ready to click "Run All"! 🚀

---

**Good luck with your training! The P100 will crush through your dataset in no time. 💪**

