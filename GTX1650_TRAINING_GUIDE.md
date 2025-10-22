# GTX 1650 Training Guide

## ✅ Great News!

Your **GTX 1650 GPU can train these models!** Much better than CPU.

---

## Training Time with GTX 1650

| Task | GTX 1650 | Colab T4 | Your CPU |
|------|----------|----------|----------|
| Behavior Model | **6-10 hours** | 2-4 hours | 40-60 hours |
| Face Model | **1-2 hours** | 30-60 min | 6-10 hours |
| **Total** | **~8-12 hours** | ~3-5 hours | ~50-70 hours |

### Verdict: 
- ✅ **5-8x faster than CPU**
- ⚠️ ~2-3x slower than Colab T4 (but you can use your own machine)
- ✅ **Perfectly viable!**

---

## Setup Instructions

### Step 1: Install CUDA PyTorch

**Double-click this file:**
```
setup_gtx1650.bat
```

Or run manually:
```bash
# Uninstall CPU PyTorch
pip uninstall torch torchvision -y

# Install CUDA PyTorch for GTX 1650
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Step 2: Verify GPU Detection

```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"}')"
```

Should output:
```
CUDA: True
GPU: NVIDIA GeForce GTX 1650
```

### Step 3: Free Up RAM

Close all unnecessary programs:
- ❌ Chrome/Edge (uses lots of RAM)
- ❌ Games
- ❌ Video editors
- ✅ Keep only terminal/PowerShell

Target: Get at least 2-3 GB free RAM

### Step 4: Train Models

```bash
# Train behavior model (6-10 hours)
python train_behavior_model.py

# Then train face model (1-2 hours)
python train_face_model.py

# Create face database (1 min)
python create_face_database.py
```

---

## Configuration (Already Optimized)

Models are now configured for GTX 1650's 4GB VRAM:

### Behavior Model:
- **Batch size**: 24 (fits in 4GB)
- **Dataset**: Full 191K images
- **Epochs**: 20
- **Expected accuracy**: 85-95%

### Face Model:
- **Batch size**: 48 (smaller images)
- **Dataset**: Full 759 images
- **Epochs**: 30
- **Expected accuracy**: 90-98%

---

## What to Expect During Training

### First 30 seconds:
```
Loading datasets...
Creating model...
Model: swin_tiny_patch4_window7_224
Device: cuda
```

### During training:
```
Epoch 1/20
  Batch 50/6000 | Loss: 1.234 | Acc: 0.456
  Batch 100/6000 | Loss: 1.123 | Acc: 0.512
  ...
```

### Each epoch:
- ~20-30 minutes per epoch
- Auto-saves best model
- Can stop and resume

### GPU Usage:
- ~85-95% GPU utilization
- ~3.5-3.8 GB VRAM used
- GPU temperature: 70-80°C (normal)

---

## Monitoring

### Check GPU Usage:
```bash
# Windows Task Manager
Ctrl + Shift + Esc → Performance → GPU
```

### Check Training Progress:
```bash
python check_training_progress.py
```

### View Current Accuracy:
```bash
# In another terminal
python -c "import json; print(json.load(open('models/behavior/training_history.json'))['val_acc'][-1])"
```

---

## If You Run Out of VRAM

If you see:
```
RuntimeError: CUDA out of memory
```

Reduce batch size in `train_behavior_model.py`:
```python
'batch_size': 16,  # down from 24
```

Or even:
```python
'batch_size': 12,  # minimum for good performance
```

---

## GTX 1650 vs Colab: Pros & Cons

### GTX 1650 (Your PC):
✅ **Pros:**
- Can run anytime
- No upload/download needed
- Full control
- No time limits
- Can pause/resume easily

❌ **Cons:**
- 2-3x slower than Colab
- Uses your PC (can't use for other tasks)
- ~8-12 hours total

### Google Colab (Cloud):
✅ **Pros:**
- 2-3x faster (T4 GPU)
- Free PC for other tasks
- ~3-5 hours total

❌ **Cons:**
- Need to upload datasets (time consuming)
- May disconnect
- Limited to 12 hour sessions
- Need internet

---

## My Recommendation for You

### Option A: Train on GTX 1650 Overnight
**Best if:**
- You don't want to upload datasets
- Can leave PC on overnight
- Want to learn the process

**Steps:**
1. Run `setup_gtx1650.bat`
2. Close all programs
3. Before bed: `python train_behavior_model.py`
4. Next morning: Check progress, start face model
5. Total: 1-2 nights

### Option B: Use Google Colab
**Best if:**
- Need it done faster
- Want best accuracy
- PC needed for other work

**Trade-off**: Takes time to upload datasets

---

## Final Setup Checklist

Before training:

- [ ] Install CUDA PyTorch (`setup_gtx1650.bat`)
- [ ] Verify GPU detected
- [ ] Close unnecessary programs
- [ ] Check free RAM (need 2-3 GB)
- [ ] Ensure no pending Windows updates
- [ ] Plug in laptop (if applicable)
- [ ] Good ventilation for GPU cooling

---

## Training Schedule Example

### Night 1 (before bed):
```bash
python train_behavior_model.py
```
Sleep 6-10 hours, training continues

### Morning (next day):
Check results:
```bash
python check_training_progress.py
```

### Afternoon/Evening:
```bash
python train_face_model.py  # 1-2 hours
python create_face_database.py  # 1 min
```

### Done!
Test inference:
```bash
python integrated_inference.py classroom.jpg
```

---

## Expected Results

### Behavior Model:
- **Training time**: 6-10 hours
- **Validation accuracy**: 85-95%
- **Model size**: ~110 MB
- **Works well** on all 7 behaviors

### Face Model:
- **Training time**: 1-2 hours
- **Verification accuracy**: 90-98%
- **Model size**: ~90 MB
- **Can identify** all 65 students

---

## Bottom Line

**GTX 1650 is perfectly capable!** 

**Training time**: ~8-12 hours total (overnight + few hours next day)

**vs Colab**: 2-3x slower but no hassle with uploads

**My suggestion**: Try GTX 1650 first. If you're impatient, use Colab.

Ready to start?
```bash
setup_gtx1650.bat
```

Then:
```bash
python train_behavior_model.py
```

