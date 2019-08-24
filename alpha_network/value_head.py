import torch.nn as nn
import torch.nn.functional as functional


class ValueHead(nn.Module):
    def __init__(self, num_channels, input_size):
        super(ValueHead, self).__init__()

        NUM_INTERMIDATE_CHANNELS = 1

        self.conv = nn.Conv2d(num_channels, NUM_INTERMIDATE_CHANNELS, kernel_size=1)
        self.bn = nn.BatchNorm2d(NUM_INTERMIDATE_CHANNELS)
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        x = x.view(1, -1)
        x = F.relu(self.fc1(x))
        return F.tanh(self.fc2(x))
