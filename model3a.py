"""
Model 3a: MobileNet-inspired MNIST model (depthwise-separable + GAP)
Target: ≥99.4% consistent in last 3 epochs, ≤15 epochs, <8000 params
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, stride: int = 1, p: int = 1):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, kernel_size=k, stride=stride, padding=p, groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dw(x)
        x = self.pw(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class Model_3a(nn.Module):
    def __init__(self, dropout: float = 0.05):
        super().__init__()
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(1, 12, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(12),
            nn.ReLU(inplace=True)
        )
        # Block 1
        self.ds1 = DepthwiseSeparableConv(12, 20)   # 28x28
        self.ds2 = DepthwiseSeparableConv(20, 20)
        self.pool1 = nn.MaxPool2d(2)  # 14x14
        self.do1 = nn.Dropout(dropout)

        # Block 2
        self.ds3 = DepthwiseSeparableConv(20, 28)
        self.ds4 = DepthwiseSeparableConv(28, 28)
        self.pool2 = nn.MaxPool2d(2)  # 7x7
        self.do2 = nn.Dropout(dropout)

        # Head
        self.ds5 = DepthwiseSeparableConv(28, 32)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(32, 10, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.ds1(x)
        x = self.ds2(x)
        x = self.pool1(x)
        x = self.do1(x)
        x = self.ds3(x)
        x = self.ds4(x)
        x = self.pool2(x)
        x = self.do2(x)
        x = self.ds5(x)
        x = self.gap(x)
        x = self.fc(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=1)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    m = Model_3a()
    print("Model 3a params:", count_parameters(m))
    y = m(torch.randn(1, 1, 28, 28))
    print("Output:", y.shape)


