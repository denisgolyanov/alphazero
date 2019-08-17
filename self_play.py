from interfaces.game_engine import GameEngine
from interfaces.prediction_network import PredictionNetwork
from mcts.simulator import MCTSSimulator
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

    def play(self):
        """
        Execute an entire self play game.
        :return: the identity of the winning player (GameState.PLAYER_ONE or GameState.PLAYER_TWO) or 0 if game is tied
        """
        logging.info("Starting self play game")

        game_state = self.game_engine.create_new_game()

        while not game_state.game_over():
            logging.info("\r\n%s", str(game_state))

            simulator = MCTSSimulator(self.network, game_state)
            next_action = simulator.compute_next_action()
            logging.info("Suggested action: %s", str(next_action))
            game_state = game_state.do_action(next_action)

        logging.info("\r\n%s", str(game_state))
        logging.info("Player: %s, game value: %s", str(game_state.get_player()), str(game_state.get_game_score()))

        return game_state.get_game_score()
