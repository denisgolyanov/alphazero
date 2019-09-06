from torch import nn

from alpha_network.residual_block import ResidualBlock
from torch import functional


class PolicyHead(nn.Module):
    def __init__(self, num_channels, input_size, num_possible_actions):
        super(PolicyHead, self).__init__()

        NUM_INTERMIDATE_CHANNELS = 2

        self.conv = nn.Conv2d(num_channels, NUM_INTERMIDATE_CHANNELS, kernel_size=1)
        self.bn = nn.BatchNorm2d(NUM_INTERMIDATE_CHANNELS)
        self.fc = nn.Linear(input_size * NUM_INTERMIDATE_CHANNELS, num_possible_actions)
        self.logSoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = functional.F.relu(self.bn(self.conv(x)))
        x = x.view(x.size()[0], -1)
        x = self.fc(x)
        x = self.logSoftmax(x)

        return x