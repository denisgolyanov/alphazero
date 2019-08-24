from torch import nn

from alpha_network.residual_block import ResidualBlock
from torch import functional


class FeatureExtractor(nn.Module):
    def __init__(self, residual_depth, input_channels, output_channels):

        super(FeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, output_channels, stride=1,
                               kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(output_channels)

        self.residual_blocks = [ResidualBlock(output_channels) for i in range(residual_depth)]

    def forward(self, x):
        x = functional.relu(self.bn1(self.conv1(x)))
        for residual_block in self.residual_blocks:
            x = residual_block.forward(x)
        return x
