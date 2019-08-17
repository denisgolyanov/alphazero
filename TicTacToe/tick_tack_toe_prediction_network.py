from interfaces.prediction_network import PredictionNetwork
from TicTacToe.tick_tack_toe_state import TickTackToeState

import random


class TickTackToePredictionNetwork(PredictionNetwork):
    def predict(self, state):
        """
        :param state: The TickTackToeState to predict for
        :return: a pair (action_probability_pairs, value)
                 where action_probability_pairs is a list of (action, probability) pairs predicted by the network
                 the value is the probability that the current player will win the game, predicted by the network
        """
        assert isinstance(state, TickTackToeState)
        all_possible_actions = state.all_possible_actions()
        action_probability_pairs = [(action, random.random()) for action in all_possible_actions]
        return action_probability_pairs, (random.random() * 2 - 1)  # U[-1, 1]
