"""
Create Face Embeddings Database
Pre-computes embeddings for all students in NDB/ directory
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision import transforms
from typing import Dict, List
import json

from train_face_model import FaceEmbeddingModel


def create_face_database(
    ndb_dir: str = 'NDB',
    model_path: str = 'models/face/best_face_model.pth',
    output_path: str = 'face_embeddings.npy',
    device: str = 'cuda'
):
    """
    Create face embeddings database from NDB directory
    
    Args:
        ndb_dir: Path to NDB directory
        model_path: Path to trained face model
        output_path: Where to save embeddings database
        device: 'cuda' or 'cpu'
    """
    print("="*70)
    print("CREATING FACE EMBEDDINGS DATABASE")
    print("="*70)
    
    # Load model
    print(f"\nLoading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device)
    embedding_size = checkpoint.get('embedding_size', 512)
    
    model = FaceEmbeddingModel(embedding_size=embedding_size, backbone='resnet50', pretrained=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"[OK] Model loaded (Embedding size: {embedding_size})")
    
    # Image transform
    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Collect all student images
    ndb_path = Path(ndb_dir)
    student_dirs = sorted([d for d in ndb_path.iterdir() if d.is_dir()])
    
    print(f"\nFound {len(student_dirs)} students in {ndb_dir}")
    
    all_embeddings = []
    all_names = []
    all_ids = []
    
    # Process each student
    for student_dir in student_dirs:
        student_full_name = student_dir.name
        student_id = student_full_name.split('_')[0]  # Extract ID
        student_name = '_'.join(student_full_name.split('_')[1:])  # Extract name
        
        # Get all images
        image_files = list(student_dir.glob('*.jpg'))
        
        if not image_files:
            print(f"  [WARNING] No images found for {student_full_name}")
            continue
        
        student_embeddings = []
        
        # Extract embeddings for all images
        with torch.no_grad():
            for img_path in image_files:
                try:
                    # Load and transform image
                    img = Image.open(img_path).convert('RGB')
                    img_tensor = transform(img).unsqueeze(0).to(device)
                    
                    # Extract embedding
                    embedding = model(img_tensor)
                    embedding = F.normalize(embedding, p=2, dim=1)
                    student_embeddings.append(embedding.cpu().numpy())
                
                except Exception as e:
                    print(f"  [ERROR] Failed to process {img_path}: {e}")
        
        if student_embeddings:
            # Average embeddings for this student
            avg_embedding = np.mean(student_embeddings, axis=0)
            
            all_embeddings.append(avg_embedding[0])
            all_names.append(student_name)
            all_ids.append(student_id)
            
            print(f"  {student_id} - {student_name:<30} ({len(student_embeddings)} images)")
    
    # Convert to numpy arrays
    all_embeddings = np.array(all_embeddings)
    
    # Create database dict
    database = {
        'embeddings': all_embeddings,
        'names': all_names,
        'ids': all_ids,
        'embedding_size': embedding_size,
        'num_students': len(all_names)
    }
    
    # Save database
    np.save(output_path, database)
    
    print(f"\n{'='*70}")
    print(f"DATABASE CREATED SUCCESSFULLY")
    print(f"{'='*70}")
    print(f"Total students: {len(all_names)}")
    print(f"Embedding size: {embedding_size}")
    print(f"Database shape: {all_embeddings.shape}")
    print(f"Saved to: {output_path}")
    print(f"{'='*70}\n")
    
    # Save metadata as JSON for reference
    metadata_path = output_path.replace('.npy', '_metadata.json')
    metadata = {
        'num_students': len(all_names),
        'embedding_size': embedding_size,
        'students': [
            {'id': sid, 'name': name}
            for sid, name in zip(all_ids, all_names)
        ]
    }
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Metadata saved to: {metadata_path}\n")
    
    return database


if __name__ == '__main__':
    import sys
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    # Check if model exists
    model_path = 'models/face/best_face_model.pth'
    if not Path(model_path).exists():
        # Find any model in the directory
        model_dir = Path('models/face')
        if model_dir.exists():
            models = list(model_dir.glob('best_face_model*.pth'))
            if models:
                model_path = str(models[0])
                print(f"Using model: {model_path}")
    
    if not Path(model_path).exists():
        print(f"ERROR: Face model not found at {model_path}")
        print("Please train the face recognition model first:")
        print("  python train_face_model.py")
        sys.exit(1)
    
    # Create database
    database = create_face_database(
        ndb_dir='NDB',
        model_path=model_path,
        output_path='face_embeddings.npy',
        device=device
    )
    
    print("[DONE] Face database ready for inference!")

