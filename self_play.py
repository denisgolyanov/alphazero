from interfaces.game_engine import GameEngine
from interfaces.game_state import GameState
from interfaces.prediction_network import PredictionNetwork
from alpha_zero_agent import AlphaZeroAgent
from utils import logger, EXTREME_DEBUG_LEVEL
import logging
import torch


class TrainingExample(object):
    def __init__(self, game_state, action_probability_tensor):
        self._game_state = game_state
        self._action_probability_tesnor = action_probability_tensor
        self._value = None

    def set_value(self, game_value):
        assert self._value is None
        assert game_value in [0, GameState.PLAYER_ONE, GameState.PLAYER_TWO], "invalid value"

        if game_value == 0:
            self._value = 0.0
        elif game_value == self._game_state.get_player():
            self._value = 1.0
        else:
            self._value = -1.0

    def __str__(self):
        return str((self._game_state.convert_to_tensor(), self._action_probability_tesnor, self._value))

    def __repr__(self):
        return str(self)

    def to_tensor_tuple(self):
        value_tensor = torch.empty(1, 1, dtype=torch.double)
        value_tensor[0, 0] = self._value
        return (self._game_state.convert_to_tensor(),
                self._action_probability_tesnor,
                value_tensor)  # TODO: tensor creation should be uniform


class SelfPlay(object):
    def __init__(self, network, game_engine, num_simulation_iterations):
        """
        :param network: a PredictionNetwork object
        :param game_engine: a GameEngine object
        :param simulation_iterations: number of iterations to perform each turn
        """
        assert isinstance(network, PredictionNetwork)
        assert isinstance(game_engine, GameEngine)

        self.network = network
        self.game_engine = game_engine
        self._agent = AlphaZeroAgent(self.network, self.game_engine, num_simulation_iterations)

    def play(self):
        """
        Execute an entire self play game.
        :return: a tuple of the identity of the winning player
                (GameState.PLAYER_ONE or GameState.PLAYER_TWO) or 0 if game is tied
                and the training examples generated by the self play game, as a list of tensors.
        """
        logger.log(EXTREME_DEBUG_LEVEL, "Starting self play game")

        training_examples = list()
        game_state = self.game_engine.create_new_game()

        while not game_state.game_over():
            logger.log(EXTREME_DEBUG_LEVEL, f"\r\n{game_state}")

            next_action, mcts_probabilities_tensor = self._agent.choose_action(competitive=False,
                                                                               fetch_probabilities=True)

            training_examples.append(TrainingExample(game_state, mcts_probabilities_tensor))

            logger.log(EXTREME_DEBUG_LEVEL, f"Suggested action: {next_action}")
            game_state = game_state.do_action(next_action)

        # update training examples with the winning probability
        game_score = game_state.get_game_score()
        for training_example in training_examples:
            training_example.set_value(game_score)

        logger.log(EXTREME_DEBUG_LEVEL, f"\r\n{game_state}")
        logger.log(EXTREME_DEBUG_LEVEL, f"Player: {game_state.get_player()}, game value: {game_score}")

        return game_state.get_game_score(), [example.to_tensor_tuple() for example in training_examples]
