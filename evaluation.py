from interfaces.game_engine import GameEngine
from interfaces.game_state import GameState
from interfaces.prediction_network import PredictionNetwork
from utils import logger

from mcts.simulator import MCTSSimulator
from alpha_zero_agent import AlphaZeroAgent


class Evaluation(object):

    def __init__(self, game_engine, agentA, agentB, competitive=False):
        """
        :param game_engine: a GameEngine object
        """
        assert isinstance(game_engine, GameEngine)
        self.game_engine = game_engine
        self.agentA = agentA
        self.agentB = agentB
        self.competitive = competitive
        self.scores = {0: 0, GameState.PLAYER_ONE: 0, GameState.PLAYER_TWO: 0}

    def play(self):
        """
        Executes a single game
        :return: the identity of the winning player (GameState.PLAYER_ONE or GameState.PLAYER_TWO) or 0 if game is tied
        """
        game_state = self.game_engine.create_new_game()

        while not game_state.game_over():
            logger.verbose_debug(f"\r\n{game_state}")

            if game_state.get_player() == GameState.PLAYER_ONE:
                next_action = self.agentA.choose_action(self.competitive, game_state)
                self.agentB.notify_of_action(next_action)
            elif game_state.get_player() == GameState.PLAYER_TWO:
                next_action = self.agentB.choose_action(self.competitive, game_state)
                self.agentA.notify_of_action(next_action)
            else:
                raise Exception("Neither of players' turn")

            logger.verbose_debug(f"Suggested action: {next_action}")
            game_state = game_state.do_action(next_action)

        logger.verbose_debug(f"\r\nFinal {game_state}")
        logger.verbose_debug(f"Player: {game_state.get_player()}, game value: {game_state.get_game_score()}")

        return game_state.get_game_score()

    def play_n_games(self, number_of_games):
        assert number_of_games % 2 == 0
        for i in range(number_of_games):
            logger.debug(f"Eval round {i}/{number_of_games}")
            if i == number_of_games / 2:
                self._switch_players()

            self.scores[self.play()] += 1
            self.agentA.restart_game()
            self.agentB.restart_game()

        # switch scores again, agentB's winnings will be reported as player B
        self._switch_players()
        return self.scores

    def _switch_players(self):
        self.agentA, self.agentB = self.agentB, self.agentA

        # Move agentB's score to the first player, from now all his wins will be counted for PLAYER_ONE
        self.scores = {0:  self.scores[0],
                       GameState.PLAYER_ONE: self.scores[GameState.PLAYER_TWO],
                       GameState.PLAYER_TWO: self.scores[GameState.PLAYER_ONE]}

