"""
    This module implements the Deep Learning models used
    for the Chinese MNIST task
"""

import torch
import torch.nn as nn

class ConvBlock(nn.Module):

    def __init__(
        self,
        num_inp_channels: int,
        num_out_fmaps: int,
        kernel_size: int,
        pool_size: int=2) -> None:

        super().__init__()
        self.conv = nn.Conv2d(in_channels=num_inp_channels, out_channels=num_out_fmaps, kernel_size=(kernel_size,kernel_size))
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=(pool_size,pool_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool(self.relu(self.conv(x)))

class CNNNet(nn.Module):
   
    def __init__(self, num_classes):
        super().__init__()

        self._conv1 = ConvBlock(num_inp_channels=1,   num_out_fmaps=64,  kernel_size=3, pool_size=2)
        self._conv2 = ConvBlock(num_inp_channels=64,  num_out_fmaps=128, kernel_size=3, pool_size=2)
        self._conv3 = ConvBlock(num_inp_channels=128, num_out_fmaps=256, kernel_size=3, pool_size=2)

        self._mlp = nn.Sequential(
            nn.Linear(9216, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
            nn.LogSoftmax(dim=1))
    
    def forward(self, input):
        x = self._conv1(input)
        x = self._conv2(x)
        x = self._conv3(x)

        y = x.view(x.size(0), -1)  # Flatten the output of convolutions

        return self._mlp(y)
