import random

from mcts.simulator import MCTSSimulator

import numpy as np


class RandomAgent(object):

    def notify_of_action(self, action):
        pass

    def choose_action(self, competitive, game_state, fetch_probabilities=False):
        actions = game_state.all_possible_actions()
        return random.choice(actions)
