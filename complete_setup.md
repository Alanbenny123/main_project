# Complete Setup Guide (A-Z) - Student Behavior Analysis System

**Complete step-by-step guide to set up the entire Student Behavior Analysis System from scratch.**

---

## 📋 Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Clone Repository](#2-clone-repository)
3. [Create Virtual Environment](#3-create-virtual-environment)
4. [Install Dependencies](#4-install-dependencies)
5. [Prepare Datasets](#5-prepare-datasets)
6. [Download Models from Private GitHub Release](#6-download-models-from-private-github-release)
7. [Enroll Faces (Face Recognition Database)](#7-enroll-faces-face-recognition-database)
8. [Start Backend Server](#8-start-backend-server)
9. [Start Frontend Application](#9-start-frontend-application)
10. [Test Full System](#10-test-full-system)
11. [Performance Optimization](#11-performance-optimization)
12. [Optional: Kaggle Training](#12-optional-kaggle-training)
13. [Private Model Management](#13-private-model-management)
14. [Troubleshooting](#14-troubleshooting)
15. [Quick Reference Commands](#15-quick-reference-commands)
16. [Final Checklist](#16-final-checklist)

---

## 1. Prerequisites

**Required Software:**
- ✅ Windows 10/11
- ✅ Python 3.13.x (already installed)
- ✅ Visual C++ Build Tools (installed)
- ✅ PowerShell (pwsh)
- ✅ Node.js 18+ (for frontend)
- ✅ Git
- ✅ GitHub CLI (`gh`) - authenticated

**Verify GitHub CLI:**
```powershell
gh auth login
```

---

## 2. Clone Repository

```powershell
cd C:\Users\alanb\Downloads
git clone https://github.com/Alanbenny123/mma.git
cd mma
```

---

## 3. Create Virtual Environment

```powershell
# Create virtual environment
python -m venv .venv

# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Upgrade pip and setuptools
python -m pip install --upgrade pip setuptools wheel
```

**Note:** Always activate the virtual environment before running Python scripts:
```powershell
.\.venv\Scripts\Activate.ps1
```

---

## 4. Install Dependencies

### 4.1 Install Base Requirements

```powershell
pip install -r requirements.txt
```

### 4.2 Install Computer Vision Libraries

```powershell
pip install insightface onnxruntime ultralytics opencv-python timm torchvision
```

**Note:** If `insightface` installation fails (C++ compiler error), ensure Visual C++ Build Tools are installed.

### 4.3 Optional: Optimize ONNX Runtime (CPU)

For faster CPU inference:
```powershell
pip install onnxruntime-silicon==1.19.2 --only-binary=:all: -i https://pypi.org/simple
```

If this fails, keep the default `onnxruntime`.

---

## 5. Prepare Datasets

### 5.1 Face Recognition Dataset (NDB)

**Structure:**
```
NDB/
  NDB/
    ASI22CA001/
      image1.jpg
      image2.jpg
      ...
    ASI22CA002/
      ...
```

**Ensure:**
- ✅ Student folders are named with correct IDs (e.g., `ASI22CA001`)
- ✅ No duplicate IDs
- ✅ Images are in `.jpg`, `.jpeg`, or `.png` format
- ✅ At least 5-8 images per student

### 5.2 Behavior Dataset

The behavior dataset (`Behaviors_Features/`) should already be in the repository. No changes needed.

### 5.3 Create Model Output Directory

```powershell
New-Item -ItemType Directory -Force temptrainedoutput | Out-Null
```

---

## 6. Download Models from Private GitHub Release

### 6.1 Download Behavior Model Files

```powershell
# Download fixed behavior model
gh release download behavior-v1 `
  -R Alanbenny123/mma-artifacts `
  -p "best_behavior_model_fixed.pth" `
  -D temptrainedoutput

# Download checkpoint
gh release download behavior-v1 `
  -R Alanbenny123/mma-artifacts `
  -p "behavior_checkpoint.pth" `
  -D temptrainedoutput

# Verify downloads
Get-ChildItem temptrainedoutput
```

**Expected output:**
```
best_behavior_model_fixed.pth
behavior_checkpoint.pth
```

### 6.2 Automation Script (Optional)

Create `scripts\pull_models.ps1`:

```powershell
New-Item -ItemType Directory -Force temptrainedoutput | Out-Null

gh release download behavior-v1 `
  -R Alanbenny123/mma-artifacts `
  -p "best_behavior_model_fixed.pth" `
  -D temptrainedoutput

gh release download behavior-v1 `
  -R Alanbenny123/mma-artifacts `
  -p "behavior_checkpoint.pth" `
  -D temptrainedoutput
```

**Run:**
```powershell
pwsh -File scripts\pull_models.ps1
```

---

## 7. Enroll Faces (Face Recognition Database)

**This step creates the face recognition database from your `NDB` dataset using InsightFace Buffalo_l model.**

```powershell
.\.venv\Scripts\Activate.ps1

python enroll_all_students.py `
  --ndb_dir ".\NDB\NDB" `
  --output ".\student_faces\embeddings.pkl"
```

**What happens:**
- Loads all student images from `NDB/NDB/`
- Extracts face embeddings using InsightFace Buffalo_l
- Saves embeddings to `student_faces\embeddings.pkl`

**Expected output:**
```
✅ Enrolled 65 students
✅ Embeddings saved to student_faces\embeddings.pkl
```

**Important:**
- ✅ Fix duplicate/misnamed student IDs before enrolling
- ✅ Re-run this script if you add/remove students or update images

---

## 8. Start Backend Server

### 8.1 Kill Any Existing Backend Processes

```powershell
Get-Process -Name python -ErrorAction SilentlyContinue | Stop-Process -Force
```

### 8.2 Start Flask Backend

```powershell
.\.venv\Scripts\Activate.ps1
python app.py
```

**Expected output:**
```
✅ Behavior model loaded successfully
✅ Face recognition system initialized
 * Running on http://127.0.0.1:5000
```

### 8.3 Verify Backend is Running

**Test API status:**
```powershell
Invoke-WebRequest -Uri http://127.0.0.1:5000/api/status
```

**Expected response:**
```json
{
  "status": "ready",
  "models": {
    "behavior": "loaded",
    "face_recognition": "loaded"
  }
}
```

### 8.4 Test Video Upload (Optional)

```powershell
$File = "C:\path\to\your\video.mp4"
Invoke-WebRequest -Uri http://127.0.0.1:5000/api/predict `
  -Method Post `
  -InFile $File `
  -ContentType "video/mp4"
```

### 8.5 Change Port (Optional)

If port 5000 is busy:
```powershell
$env:PORT="5001"
python app.py
```

---

## 9. Start Frontend Application

### 9.1 Install Frontend Dependencies

```powershell
cd .\frontend
npm ci
# OR: npm install
```

### 9.2 Start Development Server

```powershell
npm run dev
```

**Expected output:**
```
  VITE v5.x.x  ready in xxx ms

  ➜  Local:   http://127.0.0.1:5173/
  ➜  Network: use --host to expose
```

### 9.3 Open Browser

Navigate to: `http://127.0.0.1:5173`

**Features:**
- 📤 Upload video or image
- 📊 View analysis results
- 🔄 Manual status refresh button (🔄)
- 👥 Student behavior breakdown

---

## 10. Test Full System

### 10.1 Prerequisites Check

Ensure all are running:
- ✅ Backend server (`http://127.0.0.1:5000`)
- ✅ Frontend server (`http://127.0.0.1:5173`)
- ✅ Face embeddings exist (`student_faces\embeddings.pkl`)
- ✅ Behavior model exists (`temptrainedoutput\best_behavior_model_fixed.pth`)

### 10.2 Test Video Upload

1. Open frontend: `http://127.0.0.1:5173`
2. Click "Upload Video" or "Upload Image"
3. Select a test video (30-60 seconds recommended for initial testing)
4. Wait for processing
5. Verify results show:
   - ✅ Detected students with names
   - ✅ Behavior counts for 7 classes:
     - `Looking_Forward`
     - `Raising_Hand`
     - `Reading`
     - `Sleeping`
     - `Standing`
     - `Turning_Around`
     - `Writting`

### 10.3 Expected Results Format

```json
{
  "students": [
    {
      "id": "ASI22CA001",
      "name": "Student Name",
      "behaviors": {
        "Looking_Forward": 45,
        "Raising_Hand": 2,
        "Reading": 10,
        ...
      }
    }
  ]
}
```

---

## 11. Performance Optimization

### 11.1 CPU Multi-threading (Already Configured)

The backend already includes these optimizations in `app.py`:

```python
import torch
torch.set_num_threads(16)
torch.set_num_interop_threads(4)
os.environ['OMP_NUM_THREADS'] = '16'
os.environ['MKL_NUM_THREADS'] = '16'
```

**Adjust based on your CPU:**
- For 8-core CPU: `set_num_threads(8)`
- For 16-core CPU: `set_num_threads(16)` (current)

### 11.2 Video Processing Tips

- **Short videos** (30-60 seconds) for quick testing
- **Long videos** (10+ minutes) work but take longer to process
- **Upload size limit:** 500MB (configured in `app.py`)

### 11.3 YOLO Detection Speed

- Already using `yolov8n` (nano) for fastest inference
- Runs on CPU by default (fast enough for most cases)

---

## 12. Optional: Kaggle Training

### 12.1 Upload Dataset to Kaggle

1. Zip your datasets:
   ```powershell
   Compress-Archive -Path .\NDB -DestinationPath .\NDB.zip
   Compress-Archive -Path .\Behaviors_Features -DestinationPath .\Behaviors_Features.zip
   ```

2. Upload to Kaggle as private datasets

### 12.2 Create Kaggle Notebook

1. Go to Kaggle → Notebooks → New Notebook
2. Enable GPU accelerator (T4 or P100)
3. Add your datasets to the notebook
4. Copy cells from `kaggle_training_notebook.ipynb`

### 12.3 Key Configuration

**Important settings:**
- `num_workers=0` (for DataLoaders)
- `DATASET_PATH = "/kaggle/input/your-dataset-name/NDB/NDB"`
- `CONFIG['device'] = torch.device('cuda')`

### 12.4 Training Configuration

```python
CONFIG = {
    'batch_size': 64,
    'num_epochs': 80,
    'learning_rate': 0.005,
    'device': 'cuda',
    'save_dir': '/kaggle/working/models'
}
```

**Regularization:**
- Label smoothing: `0.1`
- Weight decay: `1e-3`
- Dropout: `0.5` (in model)
- Multi-step LR scheduler: milestones `[27, 54]`

### 12.5 Download Trained Model

1. After training completes, download `best_behavior_model.pth`
2. Fix checkpoint keys (if needed):
   ```powershell
   python fix_behavior_checkpoint.py
   ```
3. Upload to private GitHub release:
   ```powershell
   gh release create behavior-v2 `
     temptrainedoutput\best_behavior_model_fixed.pth `
     -R Alanbenny123/mma-artifacts `
     -t "Behavior models v2" `
     -n "Updated Swin checkpoint"
   ```

---

## 13. Private Model Management

### 13.1 Create Private Repository

```powershell
gh repo create mma-artifacts --private -y
```

### 13.2 Upload Models to Release

```powershell
cd C:\Users\alanb\Downloads\mma

gh release create behavior-v1 `
  temptrainedoutput\best_behavior_model_fixed.pth `
  temptrainedoutput\behavior_checkpoint.pth `
  -R Alanbenny123/mma-artifacts `
  -t "Behavior models v1" `
  -n "Swin checkpoints (fixed + raw)"
```

### 13.3 View Releases

```powershell
gh release view behavior-v1 -R Alanbenny123/mma-artifacts --web
```

### 13.4 Add Files to Existing Release

```powershell
gh release upload behavior-v1 path\to\new_file.pth -R Alanbenny123/mma-artifacts
```

### 13.5 Download Models

```powershell
gh release download behavior-v1 `
  -R Alanbenny123/mma-artifacts `
  -p "best_behavior_model_fixed.pth" `
  -D temptrainedoutput
```

---

## 14. Troubleshooting

### 14.1 GPU Shows 0% on Kaggle

**Issue:** Kaggle UI shows 0% GPU usage but training is fast.

**Solution:** This is a known Kaggle UI bug. If epochs run in 1-2 minutes, GPU is working.

**Verify:**
```python
import torch
print(torch.cuda.is_available())  # Should be True
```

### 14.2 `num_workers` Errors on Kaggle

**Issue:** `AssertionError: can only test a child process`

**Solution:** Set `num_workers=0` in all `DataLoader`s.

### 14.3 Model Loading Error: Missing Keys

**Issue:** `RuntimeError: Missing key(s) in state_dict: 'backbone.'`

**Solution:** Run `fix_behavior_checkpoint.py` to add `backbone.` prefix to keys:

```powershell
python fix_behavior_checkpoint.py
```

### 14.4 `KeyError: 'Turning_Around'`

**Issue:** Behavior dictionary missing a class.

**Solution:** Ensure all 7 behavior classes are initialized in `face_recognition_system.py` (already fixed).

### 14.5 Frontend Crash on Results

**Issue:** Frontend crashes when displaying results.

**Solution:** Ensure `ResultsSection.jsx` matches the new response format (already updated).

### 14.6 Backend Crash During Video Processing

**Issue:** Backend crashes or hangs during video analysis.

**Solution:**
1. Check CPU threading settings in `app.py`
2. Use shorter videos for testing
3. Kill all Python processes and restart:
   ```powershell
   Get-Process -Name python -ErrorAction SilentlyContinue | Stop-Process -Force
   python app.py
   ```

### 14.7 Slow Local Processing

**Solutions:**
- ✅ Use shorter videos (30-60 seconds)
- ✅ Verify CPU threading settings
- ✅ Close other heavy applications
- ✅ Check backend logs for errors

### 14.8 Face Recognition Not Working

**Issue:** Students not detected or identified incorrectly.

**Solutions:**
1. Verify face embeddings exist: `Get-ChildItem .\student_faces\embeddings.pkl`
2. Re-enroll faces:
   ```powershell
   python enroll_all_students.py --ndb_dir ".\NDB\NDB" --output ".\student_faces\embeddings.pkl"
   ```
3. Check video quality (lighting, face visibility)

---

## 15. Quick Reference Commands

### 15.1 Kill Stray Backend Processes

```powershell
Get-Process -Name python -ErrorAction SilentlyContinue | Stop-Process -Force
```

### 15.2 Activate Virtual Environment

```powershell
.\.venv\Scripts\Activate.ps1
```

### 15.3 Re-enroll All Faces

```powershell
Remove-Item .\student_faces\embeddings.pkl -ErrorAction SilentlyContinue
python enroll_all_students.py --ndb_dir ".\NDB\NDB" --output ".\student_faces\embeddings.pkl"
```

### 15.4 Re-download Models

```powershell
pwsh -File scripts\pull_models.ps1
```

### 15.5 Check Backend Status

```powershell
Invoke-WebRequest -Uri http://127.0.0.1:5000/api/status
```

### 15.6 Start Backend

```powershell
.\.venv\Scripts\Activate.ps1
python app.py
```

### 15.7 Start Frontend

```powershell
cd .\frontend
npm run dev
```

---

## 16. Final Checklist

**Before running the system, verify:**

- [ ] Python virtual environment created and activated
- [ ] All dependencies installed (`requirements.txt` + CV libraries)
- [ ] Behavior model files downloaded to `temptrainedoutput/`
- [ ] Face embeddings created (`student_faces\embeddings.pkl`)
- [ ] Backend server running (`http://127.0.0.1:5000`)
- [ ] Frontend server running (`http://127.0.0.1:5173`)
- [ ] API status returns `"ready"`
- [ ] Test video upload works
- [ ] Results display correctly with student names and behaviors

**If all checked, you're ready to use the system! 🎉**

---

## 📚 Additional Resources

- **Backend Architecture:** `BACKEND_ARCHITECTURE.md`
- **Frontend Architecture:** `FRONTEND_ARCHITECTURE.md`
- **Kaggle Training Guide:** `KAGGLE_TRAINING_GUIDE.md`
- **Face Recognition Setup:** `FACE_RECOGNITION_SETUP.md`
- **Training Status:** `TRAINING_STATUS.md`

---

## 🆘 Need Help?

1. Check `Troubleshooting` section above
2. Review backend logs (`app.py` output)
3. Check frontend browser console (F12)
4. Verify all prerequisites are installed
5. Ensure paths match your system structure

---

**Last Updated:** 2024
**System Version:** 1.0

