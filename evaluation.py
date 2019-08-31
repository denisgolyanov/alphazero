from interfaces.game_engine import GameEngine
from interfaces.game_state import GameState
from interfaces.prediction_network import PredictionNetwork
from utils import logger, EXTREME_DEBUG_LEVEL

from mcts.simulator import MCTSSimulator
from alphazeroagent import AlphaZeroAgent


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


    def play(self, competitive=False):
        """
        Execute an entire self play game.
        :return: the identity of the winning player (GameState.PLAYER_ONE or GameState.PLAYER_TWO) or 0 if game is tied
        """
        game_state = self.game_engine.create_new_game()

        while not game_state.game_over():
            logger.log(EXTREME_DEBUG_LEVEL, f"\r\n{game_state}")

            # TODO: improve according to actual player.

            if game_state.get_player() == GameState.PLAYER_ONE:
                next_action = self.agentA.choose_action(competitive, game_state)
                self.agentB.notify_of_action(next_action)
            elif game_state.get_player() == GameState.PLAYER_TWO:
                next_action = self.agentB.choose_action(competitive, game_state)
                self.agentA.notify_of_action(next_action)
            else:
                raise Exception("Neither of players' turn")
            logger.log(EXTREME_DEBUG_LEVEL, f"Suggested action: {next_action}")
            game_state = game_state.do_action(next_action)

        logger.log(EXTREME_DEBUG_LEVEL, f"\r\n{game_state}")
        logger.log(EXTREME_DEBUG_LEVEL, f"Player: {game_state.get_player()}, game value: {game_state.get_game_score()}")

        return game_state.get_game_score()

    def play_n_games(self, number_of_games):
        assert number_of_games % 2 == 0
        for i in range(number_of_games):
            logger.debug(f"Eval round {i}")
            if i == number_of_games/2:
                self.switch_players()
            self.scores[self.play(competitive=self.competitive)] += 1
        return self.scores

    def switch_players(self):
        temp_agent = self.agentA
        self.agentA = self.agentB
        self.agentB = temp_agent
        ties, wins, loses = self.scores[0], self.scores[GameState.PLAYER_ONE], self.scores[GameState.PLAYER_TWO]
        self.scores = self.scores = {0: ties, GameState.PLAYER_ONE: loses, GameState.PLAYER_TWO: wins}