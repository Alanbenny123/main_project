"""
Check if all required packages are installed and ready for training
"""

def check_installation():
    """Check all required packages"""
    print("="*70)
    print("INSTALLATION CHECK")
    print("="*70)
    
    all_ok = True
    
    # Python version
    import sys
    print(f"\n[OK] Python: {sys.version.split()[0]}")
    
    # Core ML packages
    try:
        import torch
        print(f"[OK] PyTorch: {torch.__version__}")
        print(f"  - CUDA Available: {torch.cuda.is_available()}")
        print(f"  - Device: {'GPU (cuda)' if torch.cuda.is_available() else 'CPU'}")
        if not torch.cuda.is_available():
            print("  [!] Warning: Using CPU - training will be slower")
    except ImportError as e:
        print(f"[X] PyTorch: NOT INSTALLED")
        print(f"  Install: pip install torch torchvision")
        all_ok = False
    
    try:
        import torchvision
        print(f"[OK] torchvision: {torchvision.__version__}")
    except ImportError:
        print(f"[X] torchvision: NOT INSTALLED")
        all_ok = False
    
    try:
        import timm
        print(f"[OK] timm: {timm.__version__}")
    except ImportError:
        print(f"[X] timm: NOT INSTALLED")
        print(f"  Install: pip install timm")
        all_ok = False
    
    # Computer Vision
    try:
        import cv2
        print(f"[OK] opencv: {cv2.__version__}")
    except ImportError:
        print(f"[X] opencv: NOT INSTALLED")
        print(f"  Install: pip install opencv-python")
        all_ok = False
    
    # Data Science
    try:
        import numpy
        print(f"[OK] numpy: {numpy.__version__}")
    except ImportError:
        print(f"[X] numpy: NOT INSTALLED")
        all_ok = False
    
    try:
        import pandas
        print(f"[OK] pandas: {pandas.__version__}")
    except ImportError:
        print(f"[X] pandas: NOT INSTALLED")
        all_ok = False
    
    try:
        import sklearn
        print(f"[OK] scikit-learn: {sklearn.__version__}")
    except ImportError:
        print(f"[X] scikit-learn: NOT INSTALLED")
        all_ok = False
    
    # Visualization
    try:
        import matplotlib
        print(f"[OK] matplotlib: {matplotlib.__version__}")
    except ImportError:
        print(f"[X] matplotlib: NOT INSTALLED")
        all_ok = False
    
    try:
        import seaborn
        print(f"[OK] seaborn: {seaborn.__version__}")
    except ImportError:
        print(f"[X] seaborn: NOT INSTALLED")
        print(f"  Install: pip install seaborn")
        all_ok = False
    
    try:
        from PIL import Image
        import PIL
        print(f"[OK] Pillow: {PIL.__version__}")
    except ImportError:
        print(f"[X] Pillow: NOT INSTALLED")
        all_ok = False
    
    # Check dataset modules
    print("\n" + "="*70)
    print("CUSTOM MODULES")
    print("="*70)
    
    try:
        from behavior_dataset import BehaviorDataset
        print(f"[OK] behavior_dataset.py: OK")
    except Exception as e:
        print(f"[X] behavior_dataset.py: ERROR - {e}")
        all_ok = False
    
    try:
        from face_dataset import FaceDataset
        print(f"[OK] face_dataset.py: OK")
    except Exception as e:
        print(f"[X] face_dataset.py: ERROR - {e}")
        all_ok = False
    
    try:
        from train_behavior_model import BehaviorClassifier
        print(f"[OK] train_behavior_model.py: OK")
    except Exception as e:
        print(f"[X] train_behavior_model.py: ERROR - {e}")
        all_ok = False
    
    try:
        from train_face_model import FaceEmbeddingModel
        print(f"[OK] train_face_model.py: OK")
    except Exception as e:
        print(f"[X] train_face_model.py: ERROR - {e}")
        all_ok = False
    
    # Check datasets exist
    print("\n" + "="*70)
    print("DATASETS")
    print("="*70)
    
    from pathlib import Path
    
    behavior_dir = Path('Behaviors_Features')
    if behavior_dir.exists():
        n_classes = len([d for d in behavior_dir.iterdir() if d.is_dir()])
        print(f"[OK] Behaviors_Features/: Found ({n_classes} classes)")
    else:
        print(f"[X] Behaviors_Features/: NOT FOUND")
        all_ok = False
    
    ndb_dir = Path('NDB')
    if ndb_dir.exists():
        n_students = len([d for d in ndb_dir.iterdir() if d.is_dir()])
        print(f"[OK] NDB/: Found ({n_students} students)")
    else:
        print(f"[X] NDB/: NOT FOUND")
        all_ok = False
    
    # Summary
    print("\n" + "="*70)
    if all_ok:
        print("[OK] ALL CHECKS PASSED - READY TO TRAIN!")
        print("="*70)
        print("\nStart training with:")
        print("  python train_behavior_model.py")
    else:
        print("[X] SOME CHECKS FAILED")
        print("="*70)
        print("\nInstall missing packages:")
        print("  pip install -r requirements.txt")
    print()
    
    return all_ok


if __name__ == '__main__':
    check_installation()

