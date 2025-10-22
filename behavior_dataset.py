"""
Behavior Classification Dataset Loader
Handles Behaviors_Features/ dataset with 7 behavior classes
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import random


class BehaviorDataset(Dataset):
    """
    Dataset for student behavior classification
    
    Classes:
        0: Looking_Forward
        1: Raising_Hand
        2: Reading
        3: Sleeping
        4: Standing
        5: Turning_Around
        6: Writting
    """
    
    CLASSES = [
        'Looking_Forward',
        'Raising_Hand',
        'Reading',
        'Sleeping',
        'Standing',
        'Turning_Around',
        'Writting'
    ]
    
    def __init__(
        self,
        root_dir: str = 'Behaviors_Features',
        student_ids: List[str] = None,
        transform: Optional[transforms.Compose] = None,
        augment: bool = True,
        max_samples_per_class: Optional[int] = None
    ):
        """
        Args:
            root_dir: Path to Behaviors_Features directory
            student_ids: List of student IDs to include (e.g., ['ID1', 'ID2'])
            transform: Custom transforms (if None, uses default)
            augment: Whether to apply data augmentation
            max_samples_per_class: Limit samples per class (for balancing)
        """
        self.root_dir = Path(root_dir)
        self.student_ids = student_ids or ['ID1', 'ID2', 'ID3', 'ID4']
        self.max_samples_per_class = max_samples_per_class
        
        # Class to index mapping (must be before _load_samples)
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.CLASSES)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
        
        # Set transforms
        if transform:
            self.transform = transform
        else:
            self.transform = self._get_default_transforms(augment)
        
        # Load dataset
        self.samples = self._load_samples()
        
        print(f"Loaded {len(self.samples)} samples from {len(self.student_ids)} students")
        self._print_class_distribution()
    
    
    def _get_default_transforms(self, augment: bool) -> transforms.Compose:
        """Get default image transforms for Swin Transformer"""
        if augment:
            return transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(10),
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.1
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            return transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
    
    
    def _load_samples(self) -> List[Tuple[Path, int]]:
        """Load all image paths and labels"""
        samples = []
        
        for class_name in self.CLASSES:
            class_dir = self.root_dir / class_name
            
            if not class_dir.exists():
                print(f"Warning: {class_dir} not found, skipping")
                continue
            
            class_samples = []
            class_idx = self.class_to_idx[class_name]
            
            # Collect samples from specified student IDs
            for student_id in self.student_ids:
                student_dir = class_dir / student_id
                
                if not student_dir.exists():
                    continue
                
                # Find all PNG images in subdirectories
                png_files = list(student_dir.rglob('*.png'))
                
                for img_path in png_files:
                    class_samples.append((img_path, class_idx))
            
            # Limit samples per class if specified (for balancing)
            if self.max_samples_per_class and len(class_samples) > self.max_samples_per_class:
                class_samples = random.sample(class_samples, self.max_samples_per_class)
            
            samples.extend(class_samples)
        
        # Shuffle samples
        random.shuffle(samples)
        
        return samples
    
    
    def _print_class_distribution(self):
        """Print class distribution statistics"""
        class_counts = {}
        for _, label in self.samples:
            class_name = self.idx_to_class[label]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        print("\nClass Distribution:")
        print("-" * 50)
        for class_name in self.CLASSES:
            count = class_counts.get(class_name, 0)
            percentage = (count / len(self.samples) * 100) if self.samples else 0
            print(f"{class_name:<20} {count:>8,} ({percentage:>5.2f}%)")
        print("-" * 50)
    
    
    def __len__(self) -> int:
        return len(self.samples)
    
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get a single sample"""
        img_path, label = self.samples[idx]
        
        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return a blank image if loading fails
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    
    def get_class_weights(self) -> torch.Tensor:
        """Calculate class weights for imbalanced dataset"""
        class_counts = torch.zeros(len(self.CLASSES))
        
        for _, label in self.samples:
            class_counts[label] += 1
        
        # Calculate weights (inverse frequency)
        total_samples = len(self.samples)
        class_weights = total_samples / (len(self.CLASSES) * class_counts)
        
        # Normalize weights
        class_weights = class_weights / class_weights.sum() * len(self.CLASSES)
        
        return class_weights


def get_behavior_dataloaders(
    batch_size: int = 32,
    num_workers: int = 4,
    train_ids: List[str] = None,
    val_ids: List[str] = None,
    test_ids: List[str] = None,
    balance_classes: bool = False,
    max_samples_per_class: Optional[int] = None
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders
    
    Args:
        batch_size: Batch size
        num_workers: Number of worker processes
        train_ids: Student IDs for training (default: ['ID1', 'ID2', 'ID3'])
        val_ids: Student IDs for validation (default: ['ID4'])
        test_ids: Student IDs for testing (default: ['ID4'])
        balance_classes: Whether to limit samples per class for balancing
        max_samples_per_class: Max samples per class (only if balance_classes=True)
    
    Returns:
        (train_loader, val_loader, test_loader)
    """
    # Default ID splits
    train_ids = train_ids or ['ID1', 'ID2', 'ID3']
    val_ids = val_ids or ['ID4']
    test_ids = test_ids or ['ID4']
    
    # Determine max samples per class for balancing
    if balance_classes and max_samples_per_class is None:
        max_samples_per_class = 8000  # Reasonable limit for smallest class
    
    # Create datasets
    print("Creating Training Dataset...")
    train_dataset = BehaviorDataset(
        student_ids=train_ids,
        augment=True,
        max_samples_per_class=max_samples_per_class if balance_classes else None
    )
    
    print("\nCreating Validation Dataset...")
    val_dataset = BehaviorDataset(
        student_ids=val_ids,
        augment=False
    )
    
    print("\nCreating Test Dataset...")
    test_dataset = BehaviorDataset(
        student_ids=test_ids,
        augment=False
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
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
    
    return train_loader, val_loader, test_loader


# ============= Example Usage =============

if __name__ == '__main__':
    # Test dataset loading
    print("Testing Behavior Dataset Loader\n")
    
    # Create dataset
    dataset = BehaviorDataset(
        student_ids=['ID1'],
        augment=True,
        max_samples_per_class=1000  # Limit for testing
    )
    
    print(f"\nDataset size: {len(dataset)}")
    print(f"Number of classes: {len(dataset.CLASSES)}")
    
    # Test getting a sample
    img, label = dataset[0]
    print(f"\nSample image shape: {img.shape}")
    print(f"Sample label: {label} ({dataset.idx_to_class[label]})")
    
    # Test dataloader
    print("\n\nTesting Dataloaders...")
    train_loader, val_loader, test_loader = get_behavior_dataloaders(
        batch_size=16,
        num_workers=0,  # Use 0 for testing on Windows
        balance_classes=True,
        max_samples_per_class=1000
    )
    
    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Test batch loading
    for batch_imgs, batch_labels in train_loader:
        print(f"\nBatch shape: {batch_imgs.shape}")
        print(f"Labels shape: {batch_labels.shape}")
        print(f"Labels: {batch_labels[:5]}")
        break
    
    # Print class weights
    class_weights = dataset.get_class_weights()
    print(f"\nClass weights for loss function:")
    for i, weight in enumerate(class_weights):
        print(f"  {dataset.idx_to_class[i]:<20} {weight:.4f}")

