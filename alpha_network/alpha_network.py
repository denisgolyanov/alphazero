from torch import nn

from alpha_network.policy_head import PolicyHead
from alpha_network.feature_extractor import FeatureExtractor

from alpha_network.value_head import ValueHead


class AlphaNetwork(nn.Module):
    def __init__(self, residual_depth, single_channel_size, num_input_channels, num_possible_actions):
        super(AlphaNetwork, self).__init__()

        INTERMIDATE_RESIDUAL_NUM_CHANNELS = 256

        self.feature_extractor = FeatureExtractor(residual_depth,
                                                  num_input_channels,
                                                  INTERMIDATE_RESIDUAL_NUM_CHANNELS)
        self.value_head = ValueHead(INTERMIDATE_RESIDUAL_NUM_CHANNELS,
                                    single_channel_size)
        self.policy_extractor = PolicyHead(INTERMIDATE_RESIDUAL_NUM_CHANNELS,
                                           single_channel_size,
                                           num_possible_actions)

    def forward(self, x):
        x = self.feature_extractor.forward(x)
        return self.policy_extractor.forward(x), self.value_head.forward(x)
