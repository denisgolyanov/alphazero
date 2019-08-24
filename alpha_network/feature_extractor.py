from torch import nn

from alpha_network.residual_block import ResidualBlock
from torch.functional import F


class FeatureExtractor(nn.Module):
    def __init__(self, residual_depth, input_channels, output_channels):

        super(FeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(output_channels)

        self.residual_blocks = [ResidualBlock(output_channels).double() for i in range(residual_depth)] # TODO: extract to attribute

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        for residual_block in self.residual_blocks:
            x = residual_block.forward(x)
        return x
