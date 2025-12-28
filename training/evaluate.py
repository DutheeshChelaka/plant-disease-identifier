"""
Model Evaluation Script
=======================

Evaluates trained models on the test set with detailed metrics.

Usage:
    python training/evaluate.py --model-path models/saved/best_model.pth

Author: DutheeshChelaka
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

# Import our modules
import sys
sys.path.append(str(Path(__file__).parent.parent))

from models.plant_cnn import PlantCNN
from training.dataset import load_dataset, create_data_splits, get_transforms, PlantDiseaseDataset


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate Plant Disease Classifier')
    
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for evaluation (default: 32)')
    
    return parser.parse_args()


def load_model(checkpoint_path, device='cpu'):
    """
    Load trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint
        device: Device to load model on
        
    Returns:
        Loaded model
    """
    print(f"📥 Loading model from {checkpoint_path}...")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get number of classes from checkpoint
    num_classes = checkpoint.get('num_classes', 38)
    
    # Create model
    model = PlantCNN(num_classes=num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"✅ Model loaded successfully!")
    print(f"   Trained for {checkpoint.get('epoch', 0) + 1} epochs")
    print(f"   Validation accuracy: {checkpoint.get('val_acc', 0):.2f}%")
    
    return model

def evaluate_model(model, test_loader, device):
    """
    Evaluate model on test set.
    
    Args:
        model: Trained model
        test_loader: DataLoader for test data
        device: Device (cuda/cpu)
        
    Returns:
        dict: Dictionary with predictions, labels, and metrics
    """
    print("\n🔍 Evaluating model on test set...")
    print("=" * 60)
    
    model.eval()
    
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc='Testing')
        
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Get predictions
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            # Collect results
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            
            # Calculate accuracy
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({'acc': f'{100.*correct/total:.2f}%'})
    
    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probabilities = np.array(all_probabilities)
    
    # Calculate final accuracy
    accuracy = 100. * correct / total
    
    print(f"\n📊 Test Results:")
    print(f"   Accuracy: {accuracy:.2f}%")
    print(f"   Correct: {correct:,} / {total:,}")
    
    return {
        'predictions': all_predictions,
        'labels': all_labels,
        'probabilities': all_probabilities,
        'accuracy': accuracy
    }
    
def calculate_metrics(predictions, labels):
    """
    Calculate detailed classification metrics.
    
    Args:
        predictions: Model predictions
        labels: True labels
        
    Returns:
        dict: Dictionary with precision, recall, f1-score
    """
    from sklearn.metrics import precision_recall_fscore_support, classification_report
    
    print("\n📈 Calculating detailed metrics...")
    
    # Calculate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='weighted', zero_division=0
    )
    
    print(f"   Precision: {precision * 100:.2f}%")
    print(f"   Recall:    {recall * 100:.2f}%")
    print(f"   F1-Score:  {f1 * 100:.2f}%")
    
    return {
        'precision': precision * 100,
        'recall': recall * 100,
        'f1_score': f1 * 100
    }


def calculate_per_class_accuracy(predictions, labels, idx_to_class):
    """
    Calculate accuracy for each disease class.
    
    Args:
        predictions: Model predictions
        labels: True labels
        idx_to_class: Dictionary mapping indices to class names
        
    Returns:
        dict: Per-class accuracies
    """
    print("\n🌿 Calculating per-class accuracy...")
    
    class_accuracies = {}
    
    for class_idx in range(len(idx_to_class)):
        # Find samples of this class
        mask = labels == class_idx
        
        if mask.sum() > 0:
            # Calculate accuracy for this class
            class_correct = (predictions[mask] == labels[mask]).sum()
            class_total = mask.sum()
            class_acc = (class_correct / class_total) * 100
            
            class_name = idx_to_class[class_idx]
            class_accuracies[class_name] = class_acc
    
    # Show top 5 and bottom 5
    sorted_accs = sorted(class_accuracies.items(), key=lambda x: x[1], reverse=True)
    
    print("\n   Top 5 Classes:")
    for name, acc in sorted_accs[:5]:
        print(f"      {name}: {acc:.2f}%")
    
    print("\n   Bottom 5 Classes:")
    for name, acc in sorted_accs[-5:]:
        print(f"      {name}: {acc:.2f}%")
    
    return class_accuracies

def main():
    """Main evaluation function."""
    # Parse arguments
    args = parse_args()
    
    print("🌱 Plant Disease Classifier - Evaluation")
    print("=" * 60)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {device}")
    
    # Load dataset
    print("\n📂 Loading dataset...")
    dataset_info = load_dataset()
    
    # Create splits
    splits = create_data_splits(
        dataset_info['image_paths'],
        dataset_info['labels']
    )
    
    # Get transforms (no augmentation for testing)
    _, test_transform = get_transforms(augment=False)
    
    # Create test dataset
    test_dataset = PlantDiseaseDataset(
        splits['test']['paths'],
        splits['test']['labels'],
        transform=test_transform
    )
    
    # Create dataloader
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    print(f"✅ Test set loaded: {len(test_dataset)} images")
    
    # Load model
    model = load_model(args.model_path, device)
    
    # Evaluate
    results = evaluate_model(model, test_loader, device)
    
    # Calculate detailed metrics
    metrics = calculate_metrics(results['predictions'], results['labels'])
    
    # Per-class accuracy
    per_class_acc = calculate_per_class_accuracy(
        results['predictions'],
        results['labels'],
        dataset_info['idx_to_class']
    )
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Evaluation Summary")
    print("=" * 60)
    print(f"Test Accuracy:  {results['accuracy']:.2f}%")
    print(f"Precision:      {metrics['precision']:.2f}%")
    print(f"Recall:         {metrics['recall']:.2f}%")
    print(f"F1-Score:       {metrics['f1_score']:.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()