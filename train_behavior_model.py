"""
Training Script for Behavior Classification Model
Uses Swin Transformer for 7-class behavior recognition
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import timm
from pathlib import Path
import json
import time
from typing import Dict, Tuple
import numpy as np
import zipfile
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from behavior_dataset import get_behavior_dataloaders, BehaviorDataset


class BehaviorClassifier(nn.Module):
    """Swin Transformer for behavior classification"""
    
    def __init__(self, num_classes: int = 7, pretrained: bool = True, model_name: str = 'swin_tiny_patch4_window7_224'):
        super().__init__()
        
        # Load pretrained Swin Transformer
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes
        )
        
        self.num_classes = num_classes
        self.model_name = model_name
    
    def forward(self, x):
        return self.backbone(x)


class BehaviorTrainer:
    """Training pipeline for behavior classification"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        device: str = 'cuda',
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        use_class_weights: bool = True
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        
        # Loss function
        if use_class_weights:
            # Get class weights from dataset
            class_weights = train_loader.dataset.get_class_weights().to(device)
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
            print(f"Using weighted loss with weights: {class_weights}")
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer (AdamW for transformers)
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=3
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
        
        for batch_idx, (images, labels) in enumerate(self.train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * images.size(0)
            running_corrects += torch.sum(preds == labels.data)
            total_samples += images.size(0)
            
            # Progress
            if (batch_idx + 1) % 50 == 0:
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
        
        for images, labels in loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * images.size(0)
            running_corrects += torch.sum(preds == labels.data)
            total_samples += images.size(0)
        
        epoch_loss = running_loss / total_samples
        epoch_acc = running_corrects.double() / total_samples
        
        return epoch_loss, epoch_acc.item()
    
    
    def train(self, num_epochs: int, save_dir: str = 'models/behavior'):
        """Full training loop"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        print("\n" + "="*70)
        print("STARTING BEHAVIOR CLASSIFICATION TRAINING")
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
            self.scheduler.step(val_loss)
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
                self.best_model_path = save_dir / f'best_model_acc{val_acc:.4f}.pth'
                
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_loss': val_loss,
                    'history': self.history
                }, self.best_model_path)
                
                print(f"  [SAVED] New best model: {val_acc:.4f}")
            
            # Save checkpoint every 5 epochs
            if (epoch + 1) % 5 == 0:
                checkpoint_path = save_dir / f'checkpoint_epoch{epoch + 1}.pth'
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_loss': val_loss,
                    'history': self.history
                }, checkpoint_path)
        
        total_time = time.time() - start_time
        
        print("\n" + "="*70)
        print("TRAINING COMPLETE!")
        print("="*70)
        print(f"Total time: {total_time / 60:.2f} minutes")
        print(f"Best validation accuracy: {self.best_val_acc:.4f}")
        print(f"Best model saved to: {self.best_model_path}")
        print("="*70 + "\n")
        
        # Save training history
        history_path = save_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        # Plot training curves
        self.plot_training_curves(save_dir / 'training_curves.png')
    
    
    def plot_training_curves(self, save_path: str):
        """Plot and save training curves"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        # Loss
        axes[0].plot(epochs, self.history['train_loss'], label='Train', marker='o')
        axes[0].plot(epochs, self.history['val_loss'], label='Val', marker='s')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training & Validation Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Accuracy
        axes[1].plot(epochs, self.history['train_acc'], label='Train', marker='o')
        axes[1].plot(epochs, self.history['val_acc'], label='Val', marker='s')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Training & Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True)
        
        # Learning rate
        axes[2].plot(epochs, self.history['lr'], marker='o', color='green')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Learning Rate')
        axes[2].set_title('Learning Rate Schedule')
        axes[2].set_yscale('log')
        axes[2].grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        print(f"Training curves saved to {save_path}")
        plt.close()
    
    
    @torch.no_grad()
    def evaluate_detailed(self, loader: DataLoader, class_names: list) -> Dict:
        """Detailed evaluation with metrics per class"""
        self.model.eval()
        
        all_preds = []
        all_labels = []
        all_probs = []
        
        for images, labels in loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            outputs = self.model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
        
        # Classification report
        report = classification_report(
            all_labels,
            all_preds,
            target_names=class_names,
            output_dict=True
        )
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        
        return {
            'predictions': all_preds,
            'labels': all_labels,
            'probabilities': all_probs,
            'report': report,
            'confusion_matrix': cm
        }
    
    
    def plot_confusion_matrix(self, cm: np.ndarray, class_names: list, save_path: str):
        """Plot confusion matrix"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names
        )
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        print(f"Confusion matrix saved to {save_path}")
        plt.close()


# ============= Main Training Script =============

def main():
    """Main training function"""
    
    # Configuration
    config = {
        'batch_size': 24,  # Optimized for GTX 1650 (4GB VRAM)
        'num_workers': 4,  # Use multiple workers
        'num_epochs': 20,  # Full training
        'learning_rate': 1e-4,
        'weight_decay': 1e-4,
        'model_name': 'swin_tiny_patch4_window7_224',
        'pretrained': True,
        'use_class_weights': True,
        'balance_classes': False,  # Use full dataset
        'max_samples_per_class': None,  # Use all data
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'save_dir': 'temptrainedoutput'
    }
    
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Create dataloaders
    print("\n\nLoading Datasets...")
    train_loader, val_loader, test_loader = get_behavior_dataloaders(
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        balance_classes=config['balance_classes'],
        max_samples_per_class=config['max_samples_per_class']
    )
    
    # Create model
    print("\n\nCreating Model...")
    model = BehaviorClassifier(
        num_classes=7,
        pretrained=config['pretrained'],
        model_name=config['model_name']
    )
    
    print(f"Model: {config['model_name']}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Create trainer
    trainer = BehaviorTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=config['device'],
        learning_rate=config['learning_rate'],
        weight_decay=config['weight_decay'],
        use_class_weights=config['use_class_weights']
    )
    
    # Train
    trainer.train(
        num_epochs=config['num_epochs'],
        save_dir=config['save_dir']
    )
    
    # Evaluate on test set
    print("\n\nEvaluating on Test Set...")
    test_results = trainer.evaluate_detailed(
        test_loader,
        class_names=BehaviorDataset.CLASSES
    )
    
    # Print classification report
    print("\nClassification Report:")
    print("-" * 70)
    for class_name in BehaviorDataset.CLASSES:
        metrics = test_results['report'][class_name]
        print(f"{class_name:<20} Precision: {metrics['precision']:.4f} | "
              f"Recall: {metrics['recall']:.4f} | F1: {metrics['f1-score']:.4f}")
    
    print(f"\nOverall Accuracy: {test_results['report']['accuracy']:.4f}")
    
    # Plot confusion matrix
    save_dir = Path(config['save_dir'])
    trainer.plot_confusion_matrix(
        test_results['confusion_matrix'],
        BehaviorDataset.CLASSES,
        save_dir / 'confusion_matrix.png'
    )
    
    # Save test results
    test_results_path = save_dir / 'test_results.json'
    with open(test_results_path, 'w') as f:
        json.dump({
            'report': test_results['report'],
            'confusion_matrix': test_results['confusion_matrix'].tolist()
        }, f, indent=2)
    
    print(f"\nTest results saved to {test_results_path}")
    print("\n[DONE] Training and evaluation complete!")
    
    # Create archive for download
    create_download_archive(config['save_dir'])


def create_download_archive(save_dir: str):
    """Create a ZIP archive of all training outputs for download"""
    save_path = Path(save_dir)
    
    if not save_path.exists():
        print(f"\nWarning: Save directory {save_dir} not found")
        return
    
    # Create archive
    archive_path = save_path.parent / f"{save_path.name}_download.zip"
    
    print(f"\n📦 Creating download archive...")
    print(f"   Source: {save_path}")
    print(f"   Output: {archive_path}")
    
    files_added = 0
    with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file_path in save_path.rglob('*'):
            if file_path.is_file():
                # Use relative path within archive
                arcname = file_path.relative_to(save_path.parent)
                zipf.write(file_path, arcname)
                files_added += 1
                if files_added <= 10:  # Show first 10 files
                    print(f"   Added: {arcname}")
    
    if files_added > 10:
        print(f"   ... and {files_added - 10} more files")
    
    file_size_mb = archive_path.stat().st_size / (1024 * 1024)
    print(f"\n✅ Download archive created: {archive_path}")
    print(f"   Total files: {files_added}")
    print(f"   Size: {file_size_mb:.2f} MB")
    print(f"\n📥 Ready to download! Location: {archive_path.absolute()}")


if __name__ == '__main__':
    main()

