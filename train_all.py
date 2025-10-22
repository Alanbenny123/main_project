"""
Complete Training Pipeline
Trains both behavior classification and face recognition models
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd: str, description: str):
    """Run a command and print status"""
    print("\n" + "="*70)
    print(f"STEP: {description}")
    print("="*70)
    print(f"Command: {cmd}\n")
    
    result = subprocess.run(cmd, shell=True)
    
    if result.returncode != 0:
        print(f"\n[ERROR] {description} failed!")
        sys.exit(1)
    
    print(f"\n[OK] {description} completed successfully!")


def main():
    """Run complete training pipeline"""
    
    print("\n" + "="*70)
    print("COMPLETE TRAINING PIPELINE")
    print("="*70)
    print("\nThis will:")
    print("  1. Analyze datasets")
    print("  2. Train behavior classification model")
    print("  3. Train face recognition model")
    print("  4. Create face embeddings database")
    print("\nEstimated time: 4-8 hours on GPU\n")
    
    response = input("Continue? [y/N]: ")
    if response.lower() != 'y':
        print("Cancelled.")
        return
    
    # Step 1: Analyze datasets
    run_command(
        "python dataset_analyzer.py",
        "Dataset Analysis"
    )
    
    # Step 2: Train behavior model
    run_command(
        "python train_behavior_model.py",
        "Behavior Classification Training"
    )
    
    # Step 3: Train face model
    run_command(
        "python train_face_model.py",
        "Face Recognition Training"
    )
    
    # Step 4: Create face database
    run_command(
        "python create_face_database.py",
        "Face Embeddings Database Creation"
    )
    
    print("\n" + "="*70)
    print("ALL TRAINING COMPLETE!")
    print("="*70)
    print("\nModels saved to:")
    print("  - models/behavior/best_model.pth")
    print("  - models/face/best_face_model.pth")
    print("  - face_embeddings.npy")
    print("\nNext steps:")
    print("  1. Run inference: python integrated_inference.py <image_or_video>")
    print("  2. View training guide: see TRAINING_GUIDE.md")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()

