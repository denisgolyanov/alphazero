from torch import nn

from alpha_network.policy_head import PolicyHead
from alpha_network.feature_extractor import FeatureExtractor

from alpha_network.value_head import ValueHead

import torch
import numpy as np
from utils import logger


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
        return torch.sum((targets - outputs)**2) / outputs.size()[0]

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

    def train(self, train_examples, epochs, batch_size=32):
        """
        :param train_examples: train examples, a list of tuples of form:
                <state, expected policy, value>
        :param epochs: number of epochs to train
        :param batch_size: batch size to use

        :return: the loss at the end of each epoch
        """

        # copy the list, since we shuffle it in-place
        train_examples = list(train_examples)

        optimizer = torch.optim.SGD(self.parameters(), lr=0.01, momentum=0.9)
        losses = list()

        num_batches = np.floor(len(train_examples) / batch_size)
        logger.info("beginning training")
        for epoch in range(epochs):
            logger.debug(f'Epoch {epoch}')
            np.random.shuffle(train_examples)
            batch_index = 0
            epoch_loss = 0.0
            while batch_index < num_batches:
                optimizer.zero_grad()

                states, policies, values = list(zip(*train_examples[batch_index * batch_size: (batch_index + 1) * batch_size]))
                states = torch.cat(states)
                policies = torch.cat(policies)
                values = torch.cat(values)

                loss = self.forward_batch(states, policies, values)

                # compute gradient and do SGD step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_index += 1
                epoch_loss += loss.item()

            losses += [epoch_loss / num_batches]

        return losses

    def save_checkpoint(self, game_name):
        import datetime
        file_name = game_name + datetime.datetime.now().isoformat().replace(':', '_').replace('.', '_')

        logger.info(f"Storing checkpoint at {file_name}")

        torch.save({
            'state_dict': self.state_dict(),
        }, file_name)

    def load_checkpoint(self, file_name):
        logger.info(f"Loading CPU checkpoint from {file_name}")
        checkpoint = torch.load(file_name, map_location='cpu')
        self.load_state_dict(checkpoint['state_dict'])



