from interfaces.prediction_network import PredictionNetwork
from TicTacToe.tick_tack_toe_state import TickTackToeState

import torch

from alpha_network.alpha_network import AlphaNetwork


class TickTackToePredictionNetwork(PredictionNetwork):

    def __init__(self):
        residual_depth = 5
        single_channel_zize = 9
        num_input_channels = 1
        num_possible_actions = 9

        self._network = AlphaNetwork(residual_depth, single_channel_zize, num_input_channels, num_possible_actions)
        self._network = self._network.double()

    def predict(self, state):
        """
        :param state: The TickTackToeState to predict for
        :return: a pair (action_probability_pairs, value)
                 where action_probability_pairs is a list of (action, probability) pairs predicted by the network
                 the value is the probability that the current player will win the game, predicted by the network
        """
        assert isinstance(state, TickTackToeState)

        state_tensor = state.convert_to_tensor()
        action_probabilities, value = self._network.forward(state_tensor)

        # remove batch dimension
        action_probabilities = action_probabilities[0]
        value = value[0]

        all_possible_actions = state.all_possible_actions()
        all_possible_actions_raw = [(action.row, action.col) for action in all_possible_actions]

        for row in range(3):
            for col in range(3):
                if (row, col) not in all_possible_actions_raw:
                    action_probabilities[row * 3 + col] = 0

        action_probabilities = action_probabilities / sum(action_probabilities)

        action_probability_pairs = [(action, action_probabilities[action.row * 3 + action.col].item())
                                    for action in all_possible_actions]

        return action_probability_pairs, value.item()
