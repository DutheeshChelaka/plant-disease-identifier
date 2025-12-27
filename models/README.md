# Model Architecture

## PlantCNN - Custom Convolutional Neural Network

### Overview
A custom CNN built from scratch for plant disease classification.

### Architecture Details

**Input:** RGB images (3, 224, 224)

**Convolutional Blocks:**
1. **Block 1:** 3 → 64 channels (224×224 → 112×112)
2. **Block 2:** 64 → 128 channels (112×112 → 56×56)
3. **Block 3:** 128 → 256 channels, 2 layers (56×56 → 28×28)
4. **Block 4:** 256 → 512 channels, 2 layers (28×28 → 14×14)

**Classifier:**
- Global Average Pooling
- FC: 512 → 512 (Dropout 0.5)
- FC: 512 → 256 (Dropout 0.5)
- FC: 256 → 38 (Output)

### Model Statistics

- **Parameters:** 4,908,070 (~4.9M)
- **Model Size:** 18.72 MB
- **Expected Accuracy:** ~91%
- **Training Time:** ~40 min (GPU) / ~7 hours (CPU)

### Usage
```python
from models.plant_cnn import PlantCNN

# Create model
model = PlantCNN(num_classes=38)

# Forward pass
output = model(images)  # images: (batch, 3, 224, 224)
```

### Features

✅ Batch Normalization for stable training  
✅ Dropout for regularization  
✅ He initialization for ReLU  
✅ Global Average Pooling (no huge FC layers)  

### Next Steps

- [ ] Add ResNet50 transfer learning model
- [ ] Add model visualization
- [ ] Add training scripts