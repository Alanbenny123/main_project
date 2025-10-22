# ⚡ Kaggle Training Quick Reference

**One-page cheat sheet for Kaggle GPU training**

---

## 🚀 Quick Start (5 Steps)

```bash
# 1. Prepare dataset locally
cd C:\Users\alanb\Downloads\mma
Compress-Archive -Path "Behaviors_Features" -DestinationPath "Behaviors_Features.zip"

# 2. Upload to Kaggle
# → Go to kaggle.com/datasets → New Dataset → Upload ZIP

# 3. Create notebook
# → Go to kaggle.com/code → New Notebook → Upload kaggle_training_notebook.ipynb

# 4. Configure
# → Settings: GPU P100 ✅, Internet ON ✅

# 5. Run
# → Update DATASET_PATH in notebook → Run All
```

---

## ⚙️ Key Configuration

```python
CONFIG = {
    'batch_size': 64,        # ↑96 for faster, ↓32 if OOM
    'num_epochs': 25,        # ↓15 for quick test, ↑30 for best
    'learning_rate': 1e-4,   # Keep default
    'train_ids': ['ID1', 'ID2', 'ID3'],
    'val_ids': ['ID4'],
}

# Update this path!
DATASET_PATH = "/kaggle/input/your-dataset-name/Behaviors_Features"
```

---

## 📊 Performance Targets

| Metric     | Target | Time    |
|------------|--------|---------|
| Val Acc    | 80-90% | ~2 hrs  |
| Test Acc   | 78-88% | -       |
| Batch Size | 64     | P100    |

---

## 💾 Files to Download

**Main file:**
- `best_behavior_model.pth` (~110 MB) ⭐

**Optional:**
- `training_history.json`
- `training_curves.png`
- `confusion_matrix.png`

**Download from:** Output tab → Download All

---

## 🔧 Common Fixes

| Problem | Fix |
|---------|-----|
| Dataset not found | Update `DATASET_PATH`, verify in "Add Data" |
| CUDA out of memory | `batch_size: 32` |
| No GPU | Settings → Accelerator → GPU P100 |
| Can't download model | Settings → Internet → ON |
| Too slow | Check GPU enabled, `num_workers: 2` |

---

## 📥 Use Model Locally

```python
import torch
from train_behavior_model import BehaviorClassifier

model = BehaviorClassifier(num_classes=7, pretrained=False)
checkpoint = torch.load('models/behavior/best_behavior_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"Accuracy: {checkpoint['val_acc']:.2%}")
```

---

## ⏱️ Timeline

| Phase           | Time   |
|-----------------|--------|
| Setup           | 5 min  |
| Dataset load    | 3 min  |
| Training (25ep) | 2 hrs  |
| Evaluation      | 5 min  |
| **TOTAL**       | **~2.2 hrs** |

---

## ✅ Pre-Flight Checklist

- [ ] Dataset zipped
- [ ] Uploaded to Kaggle
- [ ] Notebook uploaded
- [ ] GPU P100 enabled
- [ ] Internet ON
- [ ] Path updated
- [ ] Click Run All! 🚀

---

## 🆘 Emergency Debug

```python
# Add this cell if something's wrong
print(f"GPU: {torch.cuda.is_available()}")
print(f"Path exists: {Path(DATASET_PATH).exists()}")
print(f"Classes: {[d.name for d in Path(DATASET_PATH).iterdir() if d.is_dir()]}")
```

---

**Full guide:** See `KAGGLE_TRAINING_GUIDE.md`

