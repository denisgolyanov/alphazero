from torch import nn

from alpha_network.policy_head import PolicyHead
from alpha_network.feature_extractor import FeatureExtractor

from alpha_network.value_head import ValueHead

import torch
import numpy as np
import logging

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

    def _forward(self, x):
        x = self.feature_extractor.forward(x)
        return self.policy_extractor.forward(x), self.value_head.forward(x)

    def predict(self, x):
        assert x.size()[0] == 1, 'Predict on only one state at a time'

        with torch.no_grad():
            policy, value = self._forward(x)

            # last layer is a log max, for prediction, undo log
            # remove batch dimension as-well.
            return policy.exp()[0], value[0]

    @staticmethod
    def _policy_loss(outputs, targets):
        """
        Cross-entropy loss as defined in the paper.
        :note: The outputs are expected to be log-softmax probabilities
        :return: The cross-entropy losses between the output probabilities and expected probabilities
                 averaged over the number of samples in the batch
        """
        return -torch.sum(outputs * targets) / outputs.size()[0]

    @staticmethod
    def _value_loss(outputs, targets):
        """
        :return: The square error between the outputs and target values, averaged over the sample size.
        """
        return torch.sum((targets - outputs.view(-1))**2) / outputs.size()[0]

    def forward_batch(self, states, target_policies, target_values):
        """
        :param states: the batch of states
        :param target_policies: the target policy for each state in the batch
        :param target_values: the target value for each state in the batch
        :return: the total loss for the network as defined in the paper (averaged cross entropy losses for all
                 policies summed with the averaged squared error for each value)
        """
        policies, values = self._forward(states)
        return self._policy_loss(policies, target_policies) + self._value_loss(values, target_values)

    def train(self, x, epochs, batch_size=32):
        """
        :param x: train examples, a list of tuples of form:
                <state, expected policy, value>
        :param epochs: number of epochs to train
        :param batch_size: batch size to use

        :return: the loss at the end of each epoch
        """
        optimizer = torch.optim.SGD(self.nnet.parameters(), lr=0.1, momentum=0.9)
        losses = list()

        for epoch in range(epochs):
            logging.warning(f'Epoch {epoch}')
            batch_index = 0
            while batch_index < np.ceil(len(x) / batch_size):
                optimizer.zero_grad()

                states, policies, values = list(zip(*x[batch_index * batch_size: (batch_index + 1) * batch_size]))
                states = torch.stack(states)
                policies = torch.stack(policies)
                values = torch.stack(values)

                loss = self.forward_batch(states, policies, values)

                # compute gradient and do SGD step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_index += 1
                losses += [loss.item()]

        return losses



