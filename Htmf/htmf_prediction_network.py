import random
from itertools import product
import torch

from interfaces.prediction_network import PredictionNetwork
from Htmf.htmf_state import HtmfState


class HtmfPredictionNetwork(PredictionNetwork):
    def __init__(self, network, length, penguins_per_player):
        self._network = network
        self._length = length
        self._penguins_per_player = penguins_per_player

    def predict(self, state):
        """
        :param state: The HtmfState to predict for
        :return: a pair (action_probability_pairs, value)
                 where action_probability_pairs is a list of (action, probability) pairs predicted by the network
                 the value is the probability that the current player will win the game, predicted by the network
        """
        assert isinstance(state, HtmfState)

        state_tensor = state.convert_to_tensor()
        action_probabilities, value = self._network.predict(state_tensor)

        all_possible_actions = state.all_possible_actions()
        all_possible_actions_raw = [(action.start_coords, action.end_coords)
                                    for action in all_possible_actions]

        penguins_coords = HtmfPredictionNetwork.get_ordered_peguins(all_possible_actions)
        penguins_indexes = {coords : i for i, coords in enumerate(penguins_coords)}

        action_to_index = lambda start_coords, end_coords: penguins_indexes[start_coords]*self._length**2 + end_coords[0]*self._length + end_coords[1]
        
        # eliminate illegal moves
        possible_actions_indexes = set(action_to_index(*action) for action in all_possible_actions_raw)
        for i in range(len(action_probabilities)):
            if i not in possible_actions_indexes:
                action_probabilities[i] = 0.

        # normalize
        action_probabilities = action_probabilities / sum(action_probabilities)

        # build pairs to return - (action, probability) for each possible action
        action_probability_pairs = \
            [(action, 
              action_probabilities[action_to_index(action.start_coords, action.end_coords)].item())
             for action in all_possible_actions]

        return action_probability_pairs, value.item()

    def translate_to_action_probabilities_tensor(self, action_mcts_probability_pairs):
        tensor = torch.zeros([1, (self._length**2)*self._penguins_per_player], dtype=torch.double)

        # output is one board of possible moves per penguin
        # first board always represents penguin on earlier coordinates, so need
        # to figure out which spot that is
        actions = (action for action, mcts_probability in action_mcts_probability_pairs)
        penguins_coords = HtmfPredictionNetwork.get_ordered_peguins(actions)

        for action, mcts_probability in action_mcts_probability_pairs:
            penguin_index = penguins_coords.index(action.start_coords)

            action_index = penguin_index*(self._length**2)
            action_index += action.end_coords[0] * self._length
            action_index += action.end_coords[1]

            tensor[0, action_index] = mcts_probability

        return tensor

    @staticmethod
    def get_ordered_peguins(actions):
        return sorted(set(a.start_coords for a in actions))
