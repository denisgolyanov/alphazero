from interfaces.prediction_network import PredictionNetwork
from connect4.connect_four_state import ConnectFourState

import torch


class ConnectFourPredictionNetwork(PredictionNetwork):

    def __init__(self, network, cols):
        self._network = network
        self._cols = cols

    def predict(self, state):
        """
        :param state: The ConnectFourState to predict for
        :return: a pair (action_probability_pairs, value)
                 where action_probability_pairs is a list of (action, probability) pairs predicted by the network
                 the value is the probability that the current player will win the game, predicted by the network
        """
        assert isinstance(state, ConnectFourState)

        state_tensor = state.convert_to_tensor()
        action_probabilities, value = self._network.predict(state_tensor)

        all_possible_actions = state.all_possible_actions()
        all_possible_actions_raw = [action.col for action in all_possible_actions]

        for col in range(self._cols):
            if col not in all_possible_actions_raw:
                action_probabilities[col] = 0

        action_probabilities = action_probabilities / sum(action_probabilities)

        action_probability_pairs = [(action, action_probabilities[action.col].item())
                                    for action in all_possible_actions]

        return action_probability_pairs, value.item()

    def translate_to_action_probabilities_tensor(self, action_mcts_probability_pairs):
        tensor = torch.zeros([1, self._cols], dtype=torch.double)
        for action, mcts_probability in action_mcts_probability_pairs:
            tensor[0, action.col] = mcts_probability

        return tensor
