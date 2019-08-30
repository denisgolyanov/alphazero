from interfaces.game_engine import GameEngine
from interfaces.prediction_network import PredictionNetwork
from alphazeroagent import AlphaZeroAgent
import logging


class SelfPlay(object):

    def __init__(self, network, game_engine):
        """
        :param network: a PredictionNetwork object
        :param game_engine: a GameEngine object
        """
        assert isinstance(network, PredictionNetwork)
        assert isinstance(game_engine, GameEngine)

        self.network = network
        self.game_engine = game_engine
        self._agent = AlphaZeroAgent(self.network, self.game_engine, 1000)

    def play(self):
        """
        Execute an entire self play game.
        :return: the identity of the winning player (GameState.PLAYER_ONE or GameState.PLAYER_TWO) or 0 if game is tied
        """
        logging.info("Starting self play game")

        game_state = self.game_engine.create_new_game()

        while not game_state.game_over():
            logging.info(f"\r\n{game_state}")

            next_action = self._agent.choose_action()
            logging.info(f"Suggested action: {next_action}")
            game_state = game_state.do_action(next_action)

        logging.info(f"\r\n{game_state}")
        logging.info(f"Player: {game_state.get_player()}, game value: {game_state.get_game_score()}")

        return game_state.get_game_score()
