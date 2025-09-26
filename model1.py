"""
Model 1: Basic CNN Architecture
Based on techniques from notebooks 1-3: Architectural refinement and parameter optimization

TARGET: Establish baseline with efficient parameter usage (<8000 params)
RESULT: TBD after training
ANALYSIS: TBD after training

This model focuses on:
- Efficient channel progression
- 1x1 convolutions for dimensionality reduction  
- Global Average Pooling
- Minimal parameter count while maintaining learning capacity
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model_1(nn.Module):
    """
    Basic CNN with efficient parameter usage
    Receptive Field Calculation:
    Layer 1 (3x3): RF = 3
    Layer 2 (3x3): RF = 5  
    Layer 3 (3x3): RF = 7
    MaxPool (2x2): RF = 8
    Layer 4 (3x3): RF = 12
    Layer 5 (3x3): RF = 16
    Layer 6 (3x3): RF = 20
    GAP: RF = 20 (covers most of 28x28 input)
    """
    def __init__(self):
        super(Model_1, self).__init__()
        
        # Input Block
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=0, bias=False),  # 28->26, RF=3
            nn.ReLU()
        )
        
        # Convolution Block 1
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 12, 3, padding=0, bias=False),  # 26->24, RF=5
            nn.ReLU()
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(12, 12, 3, padding=0, bias=False),  # 24->22, RF=7
            nn.ReLU()
        )
        
        # Transition Block
        self.pool1 = nn.MaxPool2d(2, 2)  # 22->11, RF=8
        self.conv4 = nn.Sequential(
            nn.Conv2d(12, 8, 1, padding=0, bias=False),  # 11->11, RF=8 (1x1 doesn't change RF)
            nn.ReLU()
        )
        
        # Convolution Block 2
        self.conv5 = nn.Sequential(
            nn.Conv2d(8, 12, 3, padding=0, bias=False),  # 11->9, RF=12
            nn.ReLU()
        )
        
        self.conv6 = nn.Sequential(
            nn.Conv2d(12, 12, 3, padding=0, bias=False),  # 9->7, RF=16
            nn.ReLU()
        )
        
        self.conv7 = nn.Sequential(
            nn.Conv2d(12, 12, 3, padding=0, bias=False),  # 7->5, RF=20
            nn.ReLU()
        )
        
        # Output Block
        self.gap = nn.AdaptiveAvgPool2d(1)  # 5->1
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
    model = Model_1()
    print(f"Model 1 Parameters: {count_parameters(model)}")
    
    # Test forward pass
    test_input = torch.randn(1, 1, 28, 28)
    output = model(test_input)
    print(f"Output shape: {output.shape}")
