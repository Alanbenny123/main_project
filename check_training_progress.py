"""
Check training progress
"""

import json
from pathlib import Path
import os

def check_behavior_progress():
    """Check behavior model training progress"""
    print("="*70)
    print("BEHAVIOR CLASSIFICATION TRAINING PROGRESS")
    print("="*70)
    
    # Check for history file
    history_file = Path('models/behavior/training_history.json')
    if history_file.exists():
        with open(history_file) as f:
            history = json.load(f)
        
        epochs_done = len(history['train_loss'])
        latest_train_acc = history['train_acc'][-1] if history['train_acc'] else 0
        latest_val_acc = history['val_acc'][-1] if history['val_acc'] else 0
        best_val_acc = max(history['val_acc']) if history['val_acc'] else 0
        
        print(f"Epochs completed: {epochs_done}")
        print(f"Latest train accuracy: {latest_train_acc:.4f}")
        print(f"Latest val accuracy: {latest_val_acc:.4f}")
        print(f"Best val accuracy: {best_val_acc:.4f}")
    else:
        print("Not started yet or training just began")
    
    # Check for model checkpoints
    model_dir = Path('models/behavior')
    if model_dir.exists():
        checkpoints = list(model_dir.glob('*.pth'))
        if checkpoints:
            print(f"\nCheckpoints found: {len(checkpoints)}")
            for cp in sorted(checkpoints):
                size_mb = cp.stat().st_size / (1024*1024)
                print(f"  - {cp.name} ({size_mb:.1f} MB)")
    else:
        print("\nNo checkpoints yet")
    
    print()

def check_face_progress():
    """Check face model training progress"""
    print("="*70)
    print("FACE RECOGNITION TRAINING PROGRESS")
    print("="*70)
    
    history_file = Path('models/face/face_training_history.json')
    if history_file.exists():
        with open(history_file) as f:
            history = json.load(f)
        
        epochs_done = len(history['train_loss'])
        latest_train_acc = history['train_acc'][-1] if history['train_acc'] else 0
        latest_val_acc = history['val_acc'][-1] if history['val_acc'] else 0
        best_val_acc = max(history['val_acc']) if history['val_acc'] else 0
        
        print(f"Epochs completed: {epochs_done}")
        print(f"Latest train accuracy: {latest_train_acc:.4f}")
        print(f"Latest val accuracy: {latest_val_acc:.4f}")
        print(f"Best val accuracy: {best_val_acc:.4f}")
    else:
        print("Not started yet")
    
    model_dir = Path('models/face')
    if model_dir.exists():
        checkpoints = list(model_dir.glob('*.pth'))
        if checkpoints:
            print(f"\nCheckpoints found: {len(checkpoints)}")
            for cp in sorted(checkpoints):
                size_mb = cp.stat().st_size / (1024*1024)
                print(f"  - {cp.name} ({size_mb:.1f} MB)")
    else:
        print("\nNo checkpoints yet")
    
    print()

def check_face_database():
    """Check if face database is created"""
    print("="*70)
    print("FACE EMBEDDINGS DATABASE")
    print("="*70)
    
    db_file = Path('face_embeddings.npy')
    if db_file.exists():
        size_mb = db_file.stat().st_size / (1024*1024)
        print(f"Status: Created ({size_mb:.2f} MB)")
    else:
        print("Status: Not created yet")
    
    print()

if __name__ == '__main__':
    check_behavior_progress()
    check_face_progress()
    check_face_database()
    
    print("="*70)
    print("To monitor in real-time, run this script periodically:")
    print("  python check_training_progress.py")
    print("="*70)

