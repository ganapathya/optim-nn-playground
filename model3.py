"""
Model 3: Optimized Architecture with Data Augmentation and LR Scheduling
Based on techniques from notebooks 7-10: Data augmentation, LR scheduling, and optimization

TARGET: Achieve 99.4% consistent accuracy in last few epochs with <8000 params in ≤15 epochs
RESULT: TBD after training
ANALYSIS: TBD after training

This model incorporates:
- Optimized channel progression for better feature learning
- Strategic dropout placement
- Enhanced depth while maintaining parameter efficiency
- Designed to work with data augmentation and LR scheduling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model_3(nn.Module):
    """
    Optimized CNN with enhanced feature learning
    Receptive Field Calculation:
    Layer 1 (3x3): RF = 3
    Layer 2 (3x3): RF = 5
    Transition 1x1: RF = 5
    MaxPool (2x2): RF = 6
    Layer 3 (3x3): RF = 10
    Layer 4 (3x3): RF = 14
    Layer 5 (3x3): RF = 18
    Layer 6 (3x3, pad=1): RF = 20
    GAP: RF = 20 (good coverage for 28x28)
    """
    def __init__(self, dropout_rate=0.08):  # Reduced dropout for better final performance
        super(Model_3, self).__init__()
        
        # Input Block - Optimized initial capacity
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 10, 3, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Dropout(dropout_rate)
        )
        
        # Convolution Block 1 - Enhanced feature extraction
        self.conv2 = nn.Sequential(
            nn.Conv2d(10, 16, 3, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_rate)
        )
        
        # Transition Block - Efficient dimensionality reduction
        self.trans1 = nn.Sequential(
            nn.Conv2d(16, 8, 1, padding=0, bias=False),  # 24->24, RF=5
            nn.ReLU(),
            nn.BatchNorm2d(8)
        )
        self.pool1 = nn.MaxPool2d(2, 2)  # 24->12, RF=6
        
        # Convolution Block 2 - Deep feature learning
        self.conv3 = nn.Sequential(
            nn.Conv2d(8, 12, 3, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(12),
            nn.Dropout(dropout_rate)
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(12, 14, 3, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(14),
            nn.Dropout(dropout_rate)
        )
        
        self.conv5 = nn.Sequential(
            nn.Conv2d(14, 14, 3, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(14),
            nn.Dropout(dropout_rate)
        )
        
        # Final feature refinement with minimal dropout
        self.conv6 = nn.Sequential(
            nn.Conv2d(14, 12, 3, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(12),
            nn.Dropout(dropout_rate * 0.25)  # minimal near head
        )
        
        # Output Block
        self.gap = nn.AdaptiveAvgPool2d(1)  # 6->1
        self.conv7 = nn.Conv2d(12, 10, 1, bias=False)  # 1->1
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.trans1(x)
        x = self.pool1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.gap(x)
        x = self.conv7(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=1)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    model = Model_3()
    print(f"Model 3 Parameters: {count_parameters(model)}")
    
    # Test forward pass
    test_input = torch.randn(1, 1, 28, 28)
    output = model(test_input)
    print(f"Output shape: {output.shape}")
