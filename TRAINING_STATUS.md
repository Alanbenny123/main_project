# Training Status

## Current Status

✅ **Setup Complete**
- Datasets analyzed (252K behavior images, 759 face images)
- Data loaders created
- Training scripts ready
- PyTorch installed (CPU version)

⚠️ **Training Not Started** 
- Need to run training (2-4 hours on CPU)

## Quick Start Training

### Option 1: Double-click to start (Windows)
```
START_TRAINING.bat
```

### Option 2: Command line
```bash
python train_behavior_model.py
```

## What Will Happen

### Training Configuration (CPU-Optimized)
- **Dataset**: 35,000 images (5,000 per class - balanced)
- **Model**: Swin Transformer Tiny (27M parameters)
- **Epochs**: 10
- **Batch Size**: 16
- **Device**: CPU
- **Time**: 2-4 hours

### Progress Monitoring
While training is running, open a new terminal and run:
```bash
python check_training_progress.py
```

This shows:
- Epochs completed
- Current accuracy
- Best validation accuracy
- Saved checkpoints

## After Behavior Training Completes

The script will save:
- `models/behavior/best_model_acc{X.XX}.pth` - Best model
- `models/behavior/training_curves.png` - Training visualization
- `models/behavior/confusion_matrix.png` - Test results
- `models/behavior/training_history.json` - Metrics

## Next Steps After Behavior Training

1. **Train Face Model** (30-60 min):
   ```bash
   python train_face_model.py
   ```

2. **Create Face Database** (1-2 min):
   ```bash
   python create_face_database.py
   ```

3. **Run Inference**:
   ```bash
   python integrated_inference.py classroom.jpg
   ```

## Training Both Models at Once

If you want to train everything in sequence:
```bash
python train_all.py
```

This will:
1. Analyze datasets
2. Train behavior model (2-4 hours)
3. Train face model (30-60 min)
4. Create face database
5. Total time: ~3-5 hours

## Hardware Note

⚠️ You're using **CPU-only PyTorch**. 

If you have an NVIDIA GPU, install CUDA version for 10x faster training:
```bash
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

Then update config back to full dataset:
- `max_samples_per_class`: None (use all data)
- `num_epochs`: 20
- `batch_size`: 32

## Files Ready to Use

- ✅ `dataset_analyzer.py` - Dataset statistics
- ✅ `behavior_dataset.py` - Data loader
- ✅ `face_dataset.py` - Data loader  
- ✅ `train_behavior_model.py` - Training script (READY TO RUN)
- ✅ `train_face_model.py` - Training script
- ✅ `integrated_inference.py` - Inference pipeline
- ✅ `check_training_progress.py` - Monitor training
- ✅ `START_TRAINING.bat` - One-click start

## Troubleshooting

### Out of Memory
Reduce batch size in `train_behavior_model.py`:
```python
'batch_size': 8  # from 16
```

### Too Slow
Already optimized for CPU. Consider:
- Install CUDA PyTorch if you have GPU
- Or wait 2-4 hours for training to complete

### Check if Training is Running
```bash
python check_training_progress.py
```

## Ready to Start?

Run this command:
```bash
python train_behavior_model.py
```

Or double-click: `START_TRAINING.bat`

Training will begin immediately!

