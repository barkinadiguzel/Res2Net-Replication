import torch
import torch.nn as nn
from layers.res2net_block import Res2NetBlock
from blocks.res2net_residual import Res2NetResidualBlock

class Res2Net(nn.Module):
    def __init__(self, num_classes=1000, layers=[3,4,6,3], s=4, in_channels=64):
        super().__init__()
        self.in_channels = in_channels

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, in_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # Layers
        self.layer1 = self._make_layer(in_channels, 64, layers[0], s=s)
        self.layer2 = self._make_layer(64, 128, layers[1], s=s, stride=2)
        self.layer3 = self._make_layer(128, 256, layers[2], s=s, stride=2)
        self.layer4 = self._make_layer(256, 512, layers[3], s=s, stride=2)

        # Classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, in_channels, out_channels, blocks, s=4, stride=1):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        layers = [Res2NetResidualBlock(in_channels, out_channels, s=s, stride=stride, downsample=downsample)]
        for _ in range(1, blocks):
            layers.append(Res2NetResidualBlock(out_channels, out_channels, s=s))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)
