import torch
import torch.nn as nn

class Res2NetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, s=4, stride=1):
        super().__init__()
        assert out_channels % s == 0, "out_channels must be divisible by s"
        self.s = s
        self.width = out_channels // s

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.convs = nn.ModuleList([
            nn.Conv2d(self.width, self.width, kernel_size=3, stride=stride, padding=1, bias=False)
            for i in range(s-1)
        ])
        self.bns = nn.ModuleList([nn.BatchNorm2d(self.width) for _ in range(s-1)])

        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))  # initial 1x1 conv
        xs = torch.chunk(x, self.s, dim=1)
        out = []

        for i in range(self.s):
            if i == 0:
                out.append(xs[i])
            else:
                y = xs[i] + out[i-1]             # hierarchical residual-like connection
                y = self.relu(self.bns[i-1](self.convs[i-1](y)))
                out.append(y)

        out = torch.cat(out, dim=1)              # concatenate all splits
        out = self.bn3(self.conv3(out))          # 1x1 conv fusion
        return self.relu(out)
