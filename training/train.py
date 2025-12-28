"""
Training Script for Plant Disease Classification
=================================================

Trains the PlantCNN model on the PlantVillage dataset.

Usage:
    python training/train.py --epochs 20 --batch-size 32

Author: DutheeshChelaka
"""

import argparse
import os
from pathlib import Path
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import our modules
import sys
sys.path.append(str(Path(__file__).parent.parent))

from models.plant_cnn import PlantCNN
from training.dataset import load_dataset, create_data_splits, get_transforms, PlantDiseaseDataset


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train Plant Disease Classifier')
    
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of training epochs (default: 20)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for training (default: 32)')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate (default: 0.001)')
    parser.add_argument('--save-dir', type=str, default='models/saved',
                       help='Directory to save models (default: models/saved)')
    
    return parser.parse_args()

def train_epoch(model, train_loader, criterion, optimizer, device, epoch, total_epochs):
    """
    Train the model for one epoch.
    
    Args:
        model: Neural network model
        train_loader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on (cuda/cpu)
        epoch: Current epoch number
        total_epochs: Total number of epochs
        
    Returns:
        tuple: (average_loss, accuracy)
    """
    model.train()  # Set model to training mode
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Progress bar
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{total_epochs} [Train]')
    
    for batch_idx, (images, labels) in enumerate(pbar):
        # Move data to device
        images = images.to(device)
        labels = labels.to(device)
        
        # Zero the gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Calculate statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{running_loss/(batch_idx+1):.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    # Calculate epoch statistics
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc

def validate(model, val_loader, criterion, device, epoch, total_epochs):
    """
    Validate the model.
    
    Args:
        model: Neural network model
        val_loader: DataLoader for validation data
        criterion: Loss function
        device: Device (cuda/cpu)
        epoch: Current epoch number
        total_epochs: Total number of epochs
        
    Returns:
        tuple: (average_loss, accuracy)
    """
    model.eval()  # Set model to evaluation mode
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Progress bar
    pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{total_epochs} [Val]  ')
    
    with torch.no_grad():  # No gradients needed for validation
        for batch_idx, (images, labels) in enumerate(pbar):
            # Move data to device
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Calculate statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{running_loss/(batch_idx+1):.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
    
    # Calculate epoch statistics
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc

def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()
    
    print("🌱 Plant Disease Classifier - Training")
    print("=" * 60)
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print("=" * 60)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️  Using device: {device}")
    
    # Load dataset
    print("\n📂 Loading dataset...")
    dataset_info = load_dataset()
    
    # Create splits
    splits = create_data_splits(
        dataset_info['image_paths'],
        dataset_info['labels']
    )
    
    # Get transforms
    train_transform, val_transform = get_transforms(augment=True)
    
    # Create datasets
    train_dataset = PlantDiseaseDataset(
        splits['train']['paths'],
        splits['train']['labels'],
        transform=train_transform
    )
    
    val_dataset = PlantDiseaseDataset(
        splits['val']['paths'],
        splits['val']['labels'],
        transform=val_transform
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    print(f"✅ Data loaded: {len(train_dataset)} train, {len(val_dataset)} val")
    
# Create model
    print("\n🧠 Creating model...")
    num_classes = dataset_info['num_classes']
    model = PlantCNN(num_classes=num_classes)
    model = model.to(device)
    
    print(f"   Parameters: {model.get_num_parameters():,}")
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Training loop
    print(f"\n🎓 Starting training for {args.epochs} epochs...")
    print("=" * 60)
    
    best_val_acc = 0.0
    start_time = time.time()
    
    for epoch in range(args.epochs):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, 
            device, epoch, args.epochs
        )
        
        # Validate
        val_loss, val_acc = validate(
            model, val_loader, criterion, 
            device, epoch, args.epochs
        )
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{args.epochs} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint_path = save_dir / 'best_model.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'num_classes': num_classes
            }, checkpoint_path)
            print(f"  ✅ Best model saved! Val Acc: {val_acc:.2f}%")
        
        print("-" * 60)
    
    # Training complete
    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print("🎉 Training Complete!")
    print(f"Total time: {total_time/60:.2f} minutes")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Model saved to: {save_dir / 'best_model.pth'}")
    print("=" * 60)


if __name__ == "__main__":
    main()