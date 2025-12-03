import torch
import torch.nn as nn
from layers.res2net_block import Res2NetBlock

class Res2NetResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, s=4, stride=1, downsample=None):
        super().__init__()
        self.res2net = Res2NetBlock(in_channels, out_channels, scale=s, stride=stride)
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.res2net(x)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)
