import torch.nn as nn
import torch.nn.functional as functional


class ResidualBlock(nn.Module):
    """
    A residual block with 2 convolutions and a skip connection
    before the last ReLU activation.
    """
    def __init__(self, num_channels):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(num_channels,
                               num_channels,
                               kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_channels)

        self.conv2 = nn.Conv2d(num_channels,
                               num_channels,
                               kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = functional.relu(self.bn1(out))

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = functional.relu(out)

        return out
