import random

from interfaces.game_agent import GameAgent


class RandomAgent(GameAgent):

    def notify_of_action(self, action):
        pass

    def choose_action(self, competitive, game_state, fetch_probabilities=False):
        actions = game_state.all_possible_actions()
        return random.choice(actions)

    def restart_game(self):
        pass
