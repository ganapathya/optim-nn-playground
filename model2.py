"""
Model 2: Enhanced Architecture with Batch Normalization and Regularization
Based on techniques from notebooks 4-6: BN, Dropout, and GAP

TARGET: Improve stability and reduce overfitting while maintaining <8000 params
RESULT: TBD after training  
ANALYSIS: TBD after training

This model adds:
- Batch Normalization for training stability
- Dropout for regularization (0.1 rate)
- Better structured architecture
- Proper activation ordering
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model_2(nn.Module):
    """
    Enhanced CNN with Batch Normalization and Dropout
    Receptive Field Calculation:
    Layer 1 (3x3): RF = 3
    Layer 2 (3x3): RF = 5
    Layer 3 (3x3): RF = 7  
    MaxPool (2x2): RF = 8
    Layer 4 (3x3): RF = 12
    Layer 5 (3x3): RF = 16
    Layer 6 (3x3): RF = 20
    Layer 7 (3x3, pad=1): RF = 22
    GAP: RF = 22 (sufficient for 28x28 input)
    """
    def __init__(self, dropout_rate=0.1):
        super(Model_2, self).__init__()
        
        # Input Block
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=0, bias=False),  # 28->26, RF=3
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Dropout(dropout_rate)
        )
        
        # Convolution Block 1
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 12, 3, padding=0, bias=False),  # 26->24, RF=5
            nn.ReLU(),
            nn.BatchNorm2d(12),
            nn.Dropout(dropout_rate)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(12, 12, 3, padding=0, bias=False),  # 24->22, RF=7
            nn.ReLU(),
            nn.BatchNorm2d(12),
            nn.Dropout(dropout_rate)
        )
        
        # Transition Block
        self.pool1 = nn.MaxPool2d(2, 2)  # 22->11, RF=8
        self.conv4 = nn.Sequential(
            nn.Conv2d(12, 8, 1, padding=0, bias=False),  # 11->11, RF=8
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Dropout(dropout_rate)
        )
        
        # Convolution Block 2  
        self.conv5 = nn.Sequential(
            nn.Conv2d(8, 12, 3, padding=0, bias=False),  # 11->9, RF=12
            nn.ReLU(),
            nn.BatchNorm2d(12),
            nn.Dropout(dropout_rate)
        )
        
        self.conv6 = nn.Sequential(
            nn.Conv2d(12, 12, 3, padding=0, bias=False),  # 9->7, RF=16
            nn.ReLU(),
            nn.BatchNorm2d(12),
            nn.Dropout(dropout_rate)
        )
        
        self.conv7 = nn.Sequential(
            nn.Conv2d(12, 12, 3, padding=1, bias=False),  # 7->7, RF=18
            nn.ReLU(),
            nn.BatchNorm2d(12),
            nn.Dropout(dropout_rate)
        )
        
        # Output Block
        self.gap = nn.AdaptiveAvgPool2d(1)  # 7->1
        self.conv8 = nn.Conv2d(12, 10, 1, bias=False)  # 1->1
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x) 
        x = self.conv3(x)
        x = self.pool1(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.gap(x)
        x = self.conv8(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=1)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    model = Model_2()
    print(f"Model 2 Parameters: {count_parameters(model)}")
    
    # Test forward pass
    test_input = torch.randn(1, 1, 28, 28)
    output = model(test_input)
    print(f"Output shape: {output.shape}")
