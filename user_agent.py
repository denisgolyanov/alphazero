import random

from interfaces.game_agent import GameAgent
from mcts.simulator import MCTSSimulator

import numpy as np


class UserAgent(GameAgent):

    def notify_of_action(self, action):
        pass

    def choose_action(self, competitive, game_state, fetch_probabilities=False):
        print(f"{game_state}")
        actions = game_state.all_possible_actions()
        [print(f"{i}: {action}") for i, action in enumerate(actions)]
        index = int(input("Choose action index "))
        print(f"choosing {actions[index]}")
        return actions[index]

    def restart_game(self):
        print(f"Restarting game")

