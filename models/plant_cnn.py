"""
Custom CNN for Plant Disease Classification
============================================

A convolutional neural network built from scratch.

Architecture:
    Input: RGB images (3, 224, 224)
    - Conv Block 1: 3 -> 64 channels
    - Conv Block 2: 64 -> 128 channels  
    - Conv Block 3: 128 -> 256 channels
    - Conv Block 4: 256 -> 512 channels
    - Global Average Pooling
    - Fully Connected Layers
    Output: 38 class predictions

Author: DutheeshChelaka
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PlantCNN(nn.Module):
    """
    Custom CNN for plant disease classification.
    
    Args:
        num_classes (int): Number of disease classes (default: 38)
        dropout_rate (float): Dropout probability (default: 0.5)
    """
    
    def __init__(self, num_classes=38, dropout_rate=0.5):
        super(PlantCNN, self).__init__()
        
        # Convolutional Block 1: 3 -> 64 channels
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 224 -> 112
        # Convolutional Block 2: 64 -> 128 channels
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 112 -> 56
        
        # Convolutional Block 3: 128 -> 256 channels
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn3_2 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # 56 -> 28
        
        # Convolutional Block 4: 256 -> 512 channels
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn4_2 = nn.BatchNorm2d(512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)  # 28 -> 14
        
        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        # Fully Connected Layers (Classifier)
        self.fc1 = nn.Linear(512, 512)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.fc3 = nn.Linear(256, num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using He initialization for ReLU."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, 3, 224, 224)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        # Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        
        # Block 3 (two conv layers)
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        
        x = self.conv3_2(x)
        x = self.bn3_2(x)
        x = F.relu(x)
        x = self.pool3(x)
        
        # Block 4 (two conv layers)
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        
        x = self.conv4_2(x)
        x = self.bn4_2(x)
        x = F.relu(x)
        x = self.pool4(x)
        
        # Global Average Pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)  # Flatten to (batch_size, 512)
        
        # Fully Connected Layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)  # Final output (no activation here)
        
        return x    
 
    def get_num_parameters(self):
        """Count total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def test_model():
    """Test the model with dummy input."""
    print("Testing PlantCNN Model...")
    print("=" * 60)
    
    # Create model
    model = PlantCNN(num_classes=38)
    
    # Create dummy input (batch of 4 images)
    dummy_input = torch.randn(4, 3, 224, 224)
    
    print(f"Input shape: {dummy_input.shape}")
    
    # Forward pass
    output = model(dummy_input)
    
    print(f"Output shape: {output.shape}")
    print(f"Number of parameters: {model.get_num_parameters():,}")
    print(f"Model size (MB): {model.get_num_parameters() * 4 / (1024**2):.2f}")
    
    print("\n✅ Model test passed!")
    print("\nModel Architecture:")
    print(model)


if __name__ == "__main__":
    test_model()   