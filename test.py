import torch
import torchvision.transforms.functional
from torch import nn


class DoubleConvolution(nn.Module):            
    def __init__(self, in_channels: int, out_channels: int):            
        super().__init__()

        self.first = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.second = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

    def forward(self, x: torch.Tensor()):        
        x = self.first(x)
        x = self.                                                                                    