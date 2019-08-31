import random

from mcts.simulator import MCTSSimulator

import numpy as np


class UserAgent(object):

    def notify_of_action(self, action):
        pass

    def choose_action(self, competitive, game_state, fetch_probabilities=False):
        print(f"{game_state}")
        actions = game_state.all_possible_actions()
        print(f"{actions}")
        index = int(input("Choose action index "))
        print(f"choosing {actions[index]}")
        return actions[index]
