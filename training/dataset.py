"""
Plant Disease Dataset Loader
=============================

PyTorch Dataset for loading and preprocessing plant disease images.

Features:
- Loads images from PlantVillage dataset
- Applies transformations (resize, normalize, augment)
- Creates train/val/test splits
- Handles class imbalance

Author: DutheeshChelaka
"""

import os
from pathlib import Path
from typing import Tuple, List, Dict
import random

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np


class PlantDiseaseDataset(Dataset):
    """
    Custom Dataset for Plant Disease images.
    
    Args:
        image_paths (list): List of paths to images
        labels (list): List of corresponding labels
        transform (callable, optional): Transformations to apply
    """
    
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        """Return the total number of images."""
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """
        Load and return a single image with its label.
        
        Args:
            idx (int): Index of the image
            
        Returns:
            tuple: (image, label)
        """
        # Load image
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
def load_dataset(data_dir='data/raw/plantvillage dataset/color', 
                 test_size=0.2, 
                 val_size=0.1, 
                 random_seed=42):
    """
    Load the PlantVillage dataset and create train/val/test splits.
    
    Args:
        data_dir (str): Path to dataset directory
        test_size (float): Fraction of data for testing (0.2 = 20%)
        val_size (float): Fraction of training data for validation
        random_seed (int): Random seed for reproducibility
        
    Returns:
        dict: Dictionary containing splits and metadata
    """
    print("📂 Loading PlantVillage dataset...")
    
    data_path = Path(data_dir)
    
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path}")
    
    # Get all class directories
    class_dirs = sorted([d for d in data_path.iterdir() if d.is_dir()])
    
    if len(class_dirs) == 0:
        raise ValueError(f"No class directories found in {data_path}")
    
    print(f"   Found {len(class_dirs)} disease classes")
    
    # Create class to index mapping
    class_to_idx = {cls.name: idx for idx, cls in enumerate(class_dirs)}
    idx_to_class = {idx: cls for cls, idx in class_to_idx.items()}    
    # Collect all image paths and labels
    image_paths = []
    labels = []
    
    for class_dir in class_dirs:
        class_name = class_dir.name
        class_idx = class_to_idx[class_name]
        
        # Get all image files (jpg, JPG, png, PNG)
        images = list(class_dir.glob('*.jpg')) + \
                 list(class_dir.glob('*.JPG')) + \
                 list(class_dir.glob('*.png')) + \
                 list(class_dir.glob('*.PNG'))
        
        for img_path in images:
            image_paths.append(str(img_path))
            labels.append(class_idx)
    
    print(f"   Total images: {len(image_paths):,}")
    
    return {
        'image_paths': image_paths,
        'labels': labels,
        'class_to_idx': class_to_idx,
        'idx_to_class': idx_to_class,
        'num_classes': len(class_to_idx)
    }

def create_data_splits(image_paths, labels, test_size=0.2, val_size=0.1, random_seed=42):
    """
    Split data into train, validation, and test sets.
    
    Args:
        image_paths (list): All image paths
        labels (list): All labels
        test_size (float): Fraction for test set
        val_size (float): Fraction of remaining for validation
        random_seed (int): Random seed
        
    Returns:
        dict: Dictionary with train/val/test splits
    """
    from sklearn.model_selection import train_test_split
    
    print("✂️  Creating train/val/test splits...")
    
    # First split: separate test set
    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        image_paths, labels,
        test_size=test_size,
        random_state=random_seed,
        stratify=labels  # Maintain class distribution
    )
    
    # Second split: separate validation from training
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_val_paths, train_val_labels,
        test_size=val_size,
        random_state=random_seed,
        stratify=train_val_labels
    )
    
    print(f"   Train: {len(train_paths):,} images ({len(train_paths)/len(image_paths)*100:.1f}%)")
    print(f"   Val:   {len(val_paths):,} images ({len(val_paths)/len(image_paths)*100:.1f}%)")
    print(f"   Test:  {len(test_paths):,} images ({len(test_paths)/len(image_paths)*100:.1f}%)")
    
    return {
        'train': {'paths': train_paths, 'labels': train_labels},
        'val': {'paths': val_paths, 'labels': val_labels},
        'test': {'paths': test_paths, 'labels': test_labels}
    }
    
def get_transforms(augment=True, img_size=224):
    """
    Get image transformations for training and validation.
    
    Args:
        augment (bool): Whether to apply data augmentation
        img_size (int): Target image size
        
    Returns:
        tuple: (train_transform, val_transform)
    """
    # ImageNet normalization stats (standard for transfer learning)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    if augment:
        # Training transforms WITH augmentation
        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(degrees=20),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    else:
        # Training transforms WITHOUT augmentation
        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    
    # Validation/Test transforms (NO augmentation)
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    return train_transform, val_transform

def test_dataset():
    """Test the dataset loading and preprocessing."""
    print("🧪 Testing PlantDiseaseDataset...")
    print("=" * 60)
    
    # Load dataset
    dataset_info = load_dataset()
    
    # Create splits
    splits = create_data_splits(
        dataset_info['image_paths'],
        dataset_info['labels']
    )
    
    # Get transforms
    train_transform, val_transform = get_transforms(augment=True)
    
    # Create dataset
    train_dataset = PlantDiseaseDataset(
        splits['train']['paths'],
        splits['train']['labels'],
        transform=train_transform
    )
    
    print(f"\n📊 Dataset Statistics:")
    print(f"   Number of classes: {dataset_info['num_classes']}")
    print(f"   Training samples: {len(train_dataset)}")
    
    # Test loading one image
    print(f"\n🖼️  Testing image loading...")
    image, label = train_dataset[0]
    print(f"   Image shape: {image.shape}")
    print(f"   Label: {label} ({dataset_info['idx_to_class'][label]})")
    print(f"   Image dtype: {image.dtype}")
    print(f"   Image range: [{image.min():.3f}, {image.max():.3f}]")
    
    # Show some class names
    print(f"\n🌿 Sample Classes:")
    for i, (class_name, idx) in enumerate(list(dataset_info['class_to_idx'].items())[:5]):
        print(f"   {idx}: {class_name}")
    print(f"   ... and {dataset_info['num_classes'] - 5} more")
    
    print("\n✅ Dataset test passed!")


if __name__ == "__main__":
    test_dataset()