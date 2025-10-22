# Google Colab Training Guide

## ⚠️ Your System Status

Based on system check:
- **RAM**: 7.5 GB total, **only 480 MB available** ❌
- **GPU**: None (CPU only) ❌
- **Training Time on Your PC**: 40-60 hours (likely to crash)

**Verdict**: ❌ **Your system cannot train these models reliably**

---

## ✅ Solution: Use Google Colab (FREE GPU)

Google Colab provides FREE GPU access - perfect for your situation!

### Benefits:
- ✅ FREE Tesla T4 GPU
- ✅ 12-16 GB RAM
- ✅ Training time: 2-4 hours (vs 40-60 hours on your CPU)
- ✅ Won't crash
- ✅ Can disconnect and check back later

---

## Setup Steps

### 1. Prepare Datasets (On Your Computer)

Compress your datasets to upload to Google Drive:

```powershell
# In PowerShell
Compress-Archive -Path Behaviors_Features -DestinationPath Behaviors_Features.zip
Compress-Archive -Path NDB -DestinationPath NDB.zip
```

### 2. Upload to Google Drive

1. Go to https://drive.google.com
2. Create a folder called `mma`
3. Upload:
   - `Behaviors_Features.zip`
   - `NDB.zip`
   - All `.py` files from your project:
     - `train_behavior_model.py`
     - `train_face_model.py`
     - `behavior_dataset.py`
     - `face_dataset.py`
     - `create_face_database.py`
     - `dataset_analyzer.py`
     - `check_installation.py`
     - `check_training_progress.py`

### 3. Open Google Colab

1. Go to https://colab.research.google.com
2. Click "New Notebook"
3. **IMPORTANT**: Click `Runtime` → `Change runtime type` → Select `GPU` (T4)

### 4. Run These Commands in Colab

Copy and paste these cells one by one:

#### Cell 1: Check GPU
```python
!nvidia-smi
```
You should see GPU info (Tesla T4 or similar)

#### Cell 2: Mount Google Drive
```python
from google.colab import drive
drive.mount('/content/drive')
```
Click the link and authorize

#### Cell 3: Install Dependencies
```python
!pip install timm seaborn -q
```

#### Cell 4: Copy Files and Extract Datasets
```python
# Adjust path if your folder is different
DRIVE_PATH = '/content/drive/MyDrive/mma'

# Copy Python scripts
!cp {DRIVE_PATH}/*.py ./

# Extract datasets
!unzip -q {DRIVE_PATH}/Behaviors_Features.zip
!unzip -q {DRIVE_PATH}/NDB.zip

# Verify
!ls -d Behaviors_Features/*/
!ls -d NDB/*/ | head -5
```

#### Cell 5: Update Config for GPU
```python
# Update behavior training config
with open('train_behavior_model.py', 'r') as f:
    content = f.read()

# Use full dataset with GPU
content = content.replace("'balance_classes': True", "'balance_classes': False")
content = content.replace("'max_samples_per_class': 5000", "'max_samples_per_class': None")
content = content.replace("'num_epochs': 10", "'num_epochs': 20")
content = content.replace("'batch_size': 16", "'batch_size': 32")

with open('train_behavior_model.py', 'w') as f:
    f.write(content)

print("✓ Config updated for GPU!")
```

#### Cell 6: Check Installation
```python
!python check_installation.py
```

#### Cell 7: Train Behavior Model (2-4 hours)
```python
!python train_behavior_model.py
```
⏰ This will take 2-4 hours. You can:
- Close the browser tab and come back later
- Let it run in background
- Check progress by re-running: `!python check_training_progress.py`

#### Cell 8: Train Face Model (30-60 min)
```python
# Update face model config
with open('train_face_model.py', 'r') as f:
    content = f.read()

content = content.replace("'backbone': 'resnet34'", "'backbone': 'resnet50'")
content = content.replace("'embedding_size': 256", "'embedding_size': 512")
content = content.replace("'num_epochs': 15", "'num_epochs': 30")
content = content.replace("'batch_size': 32", "'batch_size': 64')

with open('train_face_model.py', 'w') as f:
    f.write(content)

!python train_face_model.py
```

#### Cell 9: Create Face Database
```python
!python create_face_database.py
```

#### Cell 10: Copy Models Back to Google Drive
```python
# Save trained models to Google Drive
!cp -r models {DRIVE_PATH}/
!cp face_embeddings.npy {DRIVE_PATH}/
!cp *.json {DRIVE_PATH}/

print("\n✓ Models saved to Google Drive!")
print("Download from: Google Drive/mma/models/")
```

### 5. Download Trained Models

1. Go to Google Drive
2. Navigate to `mma/models/`
3. Download:
   - `models/behavior/best_model_acc{X.XX}.pth`
   - `models/face/best_face_model_acc{X.XX}.pth`
   - `face_embeddings.npy`
   - `face_embeddings_metadata.json`

4. Place them in your local `mma/` folder

### 6. Test Inference (On Your Computer)

```bash
python integrated_inference.py classroom.jpg
```

---

## Alternative: All-in-One Script for Colab

Create ONE cell with everything:

```python
# Setup and Training - All in One
from google.colab import drive
import os

# 1. Mount Drive
drive.mount('/content/drive')

# 2. Install deps
!pip install timm seaborn -q

# 3. Setup
DRIVE_PATH = '/content/drive/MyDrive/mma'
!cp {DRIVE_PATH}/*.py ./
!unzip -q {DRIVE_PATH}/Behaviors_Features.zip
!unzip -q {DRIVE_PATH}/NDB.zip

# 4. Update configs for GPU
for file in ['train_behavior_model.py', 'train_face_model.py']:
    with open(file, 'r') as f:
        content = f.read()
    content = content.replace("'balance_classes': True", "'balance_classes': False")
    content = content.replace("'max_samples_per_class': 5000", "'max_samples_per_class': None")
    content = content.replace("'num_epochs': 10", "'num_epochs': 20")
    content = content.replace("'num_epochs': 15", "'num_epochs': 30")
    content = content.replace("'batch_size': 16", "'batch_size': 32")
    with open(file, 'w') as f:
        f.write(content)

# 5. Train behavior model
print("\\n" + "="*60)
print("TRAINING BEHAVIOR MODEL (2-4 hours)")
print("="*60)
!python train_behavior_model.py

# 6. Train face model
print("\\n" + "="*60)
print("TRAINING FACE MODEL (30-60 min)")
print("="*60)
!python train_face_model.py

# 7. Create face database
!python create_face_database.py

# 8. Save to Drive
!cp -r models {DRIVE_PATH}/
!cp face_embeddings.* {DRIVE_PATH}/

print("\\n" + "="*60)
print("✓ ALL TRAINING COMPLETE!")
print("="*60)
print("Download models from Google Drive: mma/models/")
```

Run this one cell and wait 3-5 hours. Everything will be done automatically!

---

## Monitoring Progress

If you disconnect and want to check progress:

```python
!python check_training_progress.py
```

Or view logs:
```python
!tail -30 models/behavior/training_history.json
```

---

## Expected Results

### Behavior Classification:
- **Accuracy**: 85-95%
- **Training Time**: 2-4 hours on Colab GPU
- **Model Size**: ~110 MB

### Face Recognition:
- **Accuracy**: 90-98%
- **Training Time**: 30-60 minutes on Colab GPU  
- **Model Size**: ~90 MB

---

## Cost

**100% FREE!**

Colab provides free GPU for up to 12 hours continuous use. Your training will take ~3-5 hours total, well within limits.

---

## Questions?

1. **Can I close the browser?**
   - Yes, but the session may disconnect after ~90 minutes
   - Better to keep tab open or use Colab Pro ($10/month)

2. **What if it disconnects?**
   - Models are auto-saved every 5 epochs
   - Can resume from last checkpoint

3. **Can I use my own GPU later?**
   - Yes! After downloading models, use them on any machine
   - Just need PyTorch + the model files

---

## Summary

| Method | Time | Cost | Feasibility |
|--------|------|------|-------------|
| Your PC | 40-60h | Free | ❌ Will crash |
| **Google Colab** | **2-4h** | **Free** | ✅ **BEST** |
| Cloud GPU | 2-4h | $5-10 | ✅ If you want control |

**Go with Google Colab!** 🚀

