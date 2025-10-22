"""
Training Script for Face Recognition Model
Uses ArcFace/CosFace with ResNet backbone for student face embeddings
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from pathlib import Path
import json
import time
from typing import Dict, Tuple
import numpy as np
import math

from face_dataset import get_face_dataloaders


class ArcFaceLoss(nn.Module):
    """
    ArcFace: Additive Angular Margin Loss
    Paper: https://arxiv.org/abs/1801.07698
    """
    
    def __init__(self, embedding_size: int, num_classes: int, s: float = 30.0, m: float = 0.50):
        super().__init__()
        self.embedding_size = embedding_size
        self.num_classes = num_classes
        self.s = s  # Scale
        self.m = m  # Margin
        
        # Weight matrix (embedding_size x num_classes)
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_size))
        nn.init.xavier_uniform_(self.weight)
        
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: (batch_size, embedding_size)
            labels: (batch_size,)
        """
        # Normalize embeddings and weights
        embeddings_norm = F.normalize(embeddings, p=2, dim=1)
        weight_norm = F.normalize(self.weight, p=2, dim=1)
        
        # Cosine similarity
        cosine = F.linear(embeddings_norm, weight_norm)
        
        # Calculate theta
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        
        # One-hot encode labels
        one_hot = torch.zeros(cosine.size(), device=embeddings.device)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        
        # Apply margin to target class
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        
        loss = self.criterion(output, labels)
        
        return loss, output


class FaceEmbeddingModel(nn.Module):
    """Face embedding model with ResNet backbone"""
    
    def __init__(self, embedding_size: int = 512, backbone: str = 'resnet50', pretrained: bool = True):
        super().__init__()
        
        self.embedding_size = embedding_size
        
        # Load backbone
        if backbone == 'resnet18':
            from torchvision.models import resnet18, ResNet18_Weights
            weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = resnet18(weights=weights)
            backbone_output = 512
        elif backbone == 'resnet34':
            from torchvision.models import resnet34, ResNet34_Weights
            weights = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = resnet34(weights=weights)
            backbone_output = 512
        elif backbone == 'resnet50':
            from torchvision.models import resnet50, ResNet50_Weights
            weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            self.backbone = resnet50(weights=weights)
            backbone_output = 2048
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        
        # Remove final FC layer
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        
        # Embedding layer
        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(backbone_output, embedding_size),
            nn.BatchNorm1d(embedding_size)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        embeddings = self.embedding(features)
        return embeddings


class FaceTrainer:
    """Training pipeline for face recognition"""
    
    def __init__(
        self,
        model: nn.Module,
        num_classes: int,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        device: str = 'cuda',
        learning_rate: float = 1e-3,
        weight_decay: float = 5e-4,
        embedding_size: int = 512
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.embedding_size = embedding_size
        
        # ArcFace loss
        self.criterion = ArcFaceLoss(
            embedding_size=embedding_size,
            num_classes=num_classes,
            s=30.0,
            m=0.50
        ).to(device)
        
        # Optimizer
        params = list(model.parameters()) + list(self.criterion.parameters())
        self.optimizer = optim.SGD(
            params,
            lr=learning_rate,
            momentum=0.9,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=5,
            gamma=0.5
        )
        
        # Metrics tracking
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'lr': []
        }
        
        self.best_val_acc = 0.0
        self.best_model_path = None
    
    
    def train_epoch(self) -> Tuple[float, float]:
        """Train for one epoch"""
        self.model.train()
        
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0
        
        for batch_idx, (images, labels, _) in enumerate(self.train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            embeddings = self.model(images)
            loss, outputs = self.criterion(embeddings, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * images.size(0)
            running_corrects += torch.sum(preds == labels.data)
            total_samples += images.size(0)
            
            # Progress
            if (batch_idx + 1) % 10 == 0:
                acc = running_corrects.double() / total_samples
                print(f"  Batch {batch_idx + 1}/{len(self.train_loader)} | "
                      f"Loss: {loss.item():.4f} | Acc: {acc:.4f}", end='\r')
        
        epoch_loss = running_loss / total_samples
        epoch_acc = running_corrects.double() / total_samples
        
        return epoch_loss, epoch_acc.item()
    
    
    @torch.no_grad()
    def validate(self, loader: DataLoader) -> Tuple[float, float]:
        """Validate model"""
        self.model.eval()
        
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0
        
        for images, labels, _ in loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            embeddings = self.model(images)
            loss, outputs = self.criterion(embeddings, labels)
            
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * images.size(0)
            running_corrects += torch.sum(preds == labels.data)
            total_samples += images.size(0)
        
        epoch_loss = running_loss / total_samples
        epoch_acc = running_corrects.double() / total_samples
        
        return epoch_loss, epoch_acc.item()
    
    
    def train(self, num_epochs: int, save_dir: str = 'models/face'):
        """Full training loop"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        print("\n" + "="*70)
        print("STARTING FACE RECOGNITION TRAINING")
        print("="*70)
        print(f"Device: {self.device}")
        print(f"Epochs: {num_epochs}")
        print(f"Train batches: {len(self.train_loader)}")
        print(f"Val batches: {len(self.val_loader)}")
        print(f"Save directory: {save_dir}")
        print("="*70 + "\n")
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            epoch_start = time.time()
            
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 70)
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_acc = self.validate(self.val_loader)
            
            # Update scheduler
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['lr'].append(current_lr)
            
            epoch_time = time.time() - epoch_start
            
            print(f"\n  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
            print(f"  LR: {current_lr:.6f} | Time: {epoch_time:.2f}s")
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_model_path = save_dir / f'best_face_model_acc{val_acc:.4f}.pth'
                
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'arcface_state_dict': self.criterion.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_loss': val_loss,
                    'history': self.history,
                    'embedding_size': self.embedding_size
                }, self.best_model_path)
                
                print(f"  [SAVED] New best model: {val_acc:.4f}")
        
        total_time = time.time() - start_time
        
        print("\n" + "="*70)
        print("TRAINING COMPLETE!")
        print("="*70)
        print(f"Total time: {total_time / 60:.2f} minutes")
        print(f"Best validation accuracy: {self.best_val_acc:.4f}")
        print(f"Best model saved to: {self.best_model_path}")
        print("="*70 + "\n")
        
        # Save training history
        history_path = save_dir / 'face_training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    
    @torch.no_grad()
    def extract_embeddings(self, loader: DataLoader) -> Tuple[np.ndarray, np.ndarray, list]:
        """Extract embeddings for all samples"""
        self.model.eval()
        
        all_embeddings = []
        all_labels = []
        all_names = []
        
        for images, labels, names in loader:
            images = images.to(self.device)
            
            embeddings = self.model(images)
            embeddings = F.normalize(embeddings, p=2, dim=1)
            
            all_embeddings.append(embeddings.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_names.extend(names)
        
        all_embeddings = np.vstack(all_embeddings)
        all_labels = np.concatenate(all_labels)
        
        return all_embeddings, all_labels, all_names
    
    
    def evaluate_verification(self, threshold: float = 0.5) -> Dict:
        """Evaluate face verification accuracy"""
        # Extract test embeddings
        embeddings, labels, names = self.extract_embeddings(self.test_loader)
        
        # Compute pairwise cosine similarities
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity(embeddings)
        
        # Generate positive and negative pairs
        n_samples = len(labels)
        true_positives = 0
        true_negatives = 0
        false_positives = 0
        false_negatives = 0
        
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                sim = similarities[i, j]
                same_person = (labels[i] == labels[j])
                predicted_same = (sim > threshold)
                
                if same_person and predicted_same:
                    true_positives += 1
                elif same_person and not predicted_same:
                    false_negatives += 1
                elif not same_person and predicted_same:
                    false_positives += 1
                else:
                    true_negatives += 1
        
        accuracy = (true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives)
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'threshold': threshold,
            'true_positives': true_positives,
            'true_negatives': true_negatives,
            'false_positives': false_positives,
            'false_negatives': false_negatives
        }


# ============= Main Training Script =============

def main():
    """Main training function"""
    
    # Configuration
    config = {
        'batch_size': 48,  # Optimized for GTX 1650
        'num_workers': 4,
        'num_epochs': 30,  # Full training
        'learning_rate': 1e-2,
        'weight_decay': 5e-4,
        'embedding_size': 512,  # Full embeddings
        'backbone': 'resnet50',  # Best backbone
        'pretrained': True,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'save_dir': 'models/face'
    }
    
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Create dataloaders
    print("\n\nLoading Datasets...")
    train_loader, val_loader, test_loader, num_classes = get_face_dataloaders(
        batch_size=config['batch_size'],
        num_workers=config['num_workers']
    )
    
    # Create model
    print("\n\nCreating Model...")
    model = FaceEmbeddingModel(
        embedding_size=config['embedding_size'],
        backbone=config['backbone'],
        pretrained=config['pretrained']
    )
    
    print(f"Backbone: {config['backbone']}")
    print(f"Embedding size: {config['embedding_size']}")
    print(f"Number of students: {num_classes}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer
    trainer = FaceTrainer(
        model=model,
        num_classes=num_classes,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=config['device'],
        learning_rate=config['learning_rate'],
        weight_decay=config['weight_decay'],
        embedding_size=config['embedding_size']
    )
    
    # Train
    trainer.train(
        num_epochs=config['num_epochs'],
        save_dir=config['save_dir']
    )
    
    # Evaluate verification accuracy
    print("\n\nEvaluating Face Verification...")
    verification_results = trainer.evaluate_verification(threshold=0.5)
    
    print("\nVerification Results:")
    print(f"  Accuracy:  {verification_results['accuracy']:.4f}")
    print(f"  Precision: {verification_results['precision']:.4f}")
    print(f"  Recall:    {verification_results['recall']:.4f}")
    print(f"  Threshold: {verification_results['threshold']}")
    
    # Save verification results
    save_dir = Path(config['save_dir'])
    results_path = save_dir / 'verification_results.json'
    with open(results_path, 'w') as f:
        json.dump(verification_results, f, indent=2)
    
    print(f"\nResults saved to {results_path}")
    print("\n[DONE] Training and evaluation complete!")


if __name__ == '__main__':
    main()

