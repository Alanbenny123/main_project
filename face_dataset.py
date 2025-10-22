"""
Face Recognition Dataset Loader
Handles NDB/ dataset with student face images
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import random
from collections import defaultdict


class FaceDataset(Dataset):
    """
    Dataset for face recognition/embedding training
    
    Loads student face images from NDB/ directory
    Each student has 6-25 face images
    """
    
    def __init__(
        self,
        root_dir: str = 'NDB',
        student_list: Optional[List[str]] = None,
        split: str = 'train',
        split_ratio: Tuple[float, float, float] = (0.70, 0.15, 0.15),
        transform: Optional[transforms.Compose] = None,
        augment: bool = True,
        random_seed: int = 42
    ):
        """
        Args:
            root_dir: Path to NDB directory
            student_list: Specific students to include (None = all)
            split: 'train', 'val', or 'test'
            split_ratio: (train, val, test) ratios
            transform: Custom transforms
            augment: Whether to apply augmentation (train only)
            random_seed: Random seed for reproducible splits
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.split_ratio = split_ratio
        self.random_seed = random_seed
        
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        # Set transforms
        if transform:
            self.transform = transform
        else:
            self.transform = self._get_default_transforms(augment and split == 'train')
        
        # Load dataset
        self.samples, self.student_to_idx, self.idx_to_student = self._load_samples(student_list)
        self.num_classes = len(self.student_to_idx)
        
        print(f"Loaded {len(self.samples)} {split} samples from {self.num_classes} students")
    
    
    def _get_default_transforms(self, augment: bool) -> transforms.Compose:
        """Get default image transforms for face recognition"""
        if augment:
            return transforms.Compose([
                transforms.Resize((128, 128)),
                transforms.RandomCrop(112),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(
                    brightness=0.3,
                    contrast=0.3,
                    saturation=0.2
                ),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5],
                    std=[0.5, 0.5, 0.5]
                )
            ])
        else:
            return transforms.Compose([
                transforms.Resize((112, 112)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5],
                    std=[0.5, 0.5, 0.5]
                )
            ])
    
    
    def _load_samples(self, student_list: Optional[List[str]]) -> Tuple[List, Dict, Dict]:
        """Load samples and create student mappings"""
        # Collect all student directories
        if not self.root_dir.exists():
            raise FileNotFoundError(f"NDB directory not found: {self.root_dir}")
        
        student_dirs = sorted([d for d in self.root_dir.iterdir() if d.is_dir()])
        
        if student_list:
            student_dirs = [d for d in student_dirs if d.name in student_list]
        
        # Create student to index mapping
        student_to_idx = {student_dir.name: idx for idx, student_dir in enumerate(student_dirs)}
        idx_to_student = {idx: name for name, idx in student_to_idx.items()}
        
        # Collect samples per student
        all_student_samples = defaultdict(list)
        
        for student_dir in student_dirs:
            student_name = student_dir.name
            student_idx = student_to_idx[student_name]
            
            # Get all JPG images
            jpg_files = sorted(list(student_dir.glob('*.jpg')))
            
            for img_path in jpg_files:
                all_student_samples[student_name].append((img_path, student_idx, student_name))
        
        # Split samples per student
        samples = []
        train_ratio, val_ratio, test_ratio = self.split_ratio
        
        for student_name, student_imgs in all_student_samples.items():
            n_total = len(student_imgs)
            n_train = int(n_total * train_ratio)
            n_val = int(n_total * val_ratio)
            
            # Shuffle student images
            random.shuffle(student_imgs)
            
            # Split
            if self.split == 'train':
                samples.extend(student_imgs[:n_train])
            elif self.split == 'val':
                samples.extend(student_imgs[n_train:n_train + n_val])
            else:  # test
                samples.extend(student_imgs[n_train + n_val:])
        
        # Shuffle all samples
        random.shuffle(samples)
        
        return samples, student_to_idx, idx_to_student
    
    
    def __len__(self) -> int:
        return len(self.samples)
    
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        """Get a single sample"""
        img_path, label, student_name = self.samples[idx]
        
        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            image = Image.new('RGB', (112, 112), (0, 0, 0))
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label, student_name
    
    
    def get_student_name(self, idx: int) -> str:
        """Get student name from index"""
        return self.idx_to_student[idx]
    
    
    def get_student_id_from_name(self, full_name: str) -> str:
        """Extract student ID from folder name (e.g., ASI22CA009_Alan_Benny -> ASI22CA009)"""
        return full_name.split('_')[0]


def get_face_dataloaders(
    batch_size: int = 64,
    num_workers: int = 4,
    split_ratio: Tuple[float, float, float] = (0.70, 0.15, 0.15),
    random_seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
    """
    Create train, validation, and test dataloaders for face recognition
    
    Args:
        batch_size: Batch size
        num_workers: Number of worker processes
        split_ratio: (train, val, test) split ratios
        random_seed: Random seed for reproducible splits
    
    Returns:
        (train_loader, val_loader, test_loader, num_classes)
    """
    # Create datasets
    print("Creating Face Training Dataset...")
    train_dataset = FaceDataset(
        split='train',
        split_ratio=split_ratio,
        augment=True,
        random_seed=random_seed
    )
    
    print("\nCreating Face Validation Dataset...")
    val_dataset = FaceDataset(
        split='val',
        split_ratio=split_ratio,
        augment=False,
        random_seed=random_seed
    )
    
    print("\nCreating Face Test Dataset...")
    test_dataset = FaceDataset(
        split='test',
        split_ratio=split_ratio,
        augment=False,
        random_seed=random_seed
    )
    
    num_classes = train_dataset.num_classes
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True  # Important for metric learning
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"\nNum classes (students): {num_classes}")
    
    return train_loader, val_loader, test_loader, num_classes


# ============= Example Usage =============

if __name__ == '__main__':
    # Test dataset loading
    print("Testing Face Dataset Loader\n")
    
    # Create datasets
    train_dataset = FaceDataset(
        split='train',
        augment=True
    )
    
    print(f"\nDataset size: {len(train_dataset)}")
    print(f"Number of students: {train_dataset.num_classes}")
    
    # Test getting a sample
    img, label, student_name = train_dataset[0]
    print(f"\nSample image shape: {img.shape}")
    print(f"Sample label: {label}")
    print(f"Student name: {student_name}")
    print(f"Student ID: {train_dataset.get_student_id_from_name(student_name)}")
    
    # Test dataloader
    print("\n\nTesting Dataloaders...")
    train_loader, val_loader, test_loader, num_classes = get_face_dataloaders(
        batch_size=16,
        num_workers=0  # Use 0 for testing on Windows
    )
    
    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Test batch loading
    for batch_imgs, batch_labels, batch_names in train_loader:
        print(f"\nBatch shape: {batch_imgs.shape}")
        print(f"Labels shape: {batch_labels.shape}")
        print(f"First 3 labels: {batch_labels[:3]}")
        print(f"First 3 names: {batch_names[:3]}")
        break
    
    print("\n[OK] Face dataset loader working correctly!")

