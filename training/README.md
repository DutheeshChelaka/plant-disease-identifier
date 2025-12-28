# Training Pipeline

## Overview

Data loading and preprocessing for plant disease classification.

## Dataset

**PlantVillage Dataset:**
- 54,304 images
- 38 disease classes
- 14 crop species

## Data Split

- **Training:** 72% (39,098 images)
- **Validation:** 8% (4,345 images)
- **Test:** 20% (10,861 images)

## Image Processing

### Transformations

**Training (with augmentation):**
- Resize to 224×224
- Random horizontal flip (50%)
- Random vertical flip (20%)
- Random rotation (±20°)
- Color jitter (brightness, contrast, saturation, hue)
- Normalize with ImageNet stats

**Validation/Test (no augmentation):**
- Resize to 224×224
- Normalize with ImageNet stats

### Normalization

Uses ImageNet statistics:
- Mean: [0.485, 0.456, 0.406]
- Std: [0.229, 0.224, 0.225]

## Usage
```python
from training.dataset import load_dataset, create_data_splits, get_transforms, PlantDiseaseDataset

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
```

## Classes

All 38 disease classes:
- Apple: Apple Scab, Black Rot, Cedar Rust, Healthy
- Blueberry: Healthy
- Cherry: Powdery Mildew, Healthy
- Corn: Cercospora Leaf Spot, Common Rust, Northern Leaf Blight, Healthy
- Grape: Black Rot, Esca, Leaf Blight, Healthy
- And 24 more...

## Next Steps

- [ ] Create DataLoader utilities
- [ ] Add training script
- [ ] Add evaluation metrics