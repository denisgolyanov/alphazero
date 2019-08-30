from alphazeroagent import AlphaZeroAgent
from evaluation import Evaluation
from self_play import SelfPlay
from TicTacToe.tick_tack_toe_prediction_network import TickTackToePredictionNetwork
from TicTacToe.tick_tack_toe_game_engine import TickTackToeGameEngine
from interfaces.game_state import GameState
from alpha_network.alpha_network import AlphaNetwork

import logging

logging.basicConfig(level=logging.WARN)
logging.info("Hello world")


def __main__():

    scores = {0 : 0, GameState.PLAYER_ONE: 0, GameState.PLAYER_TWO: 0}
    all_examples = []

    for i in range(10):
        game_engine = TickTackToeGameEngine()
        residual_depth = 5
        single_channel_zize = 9
        num_input_channels = 1
        num_possible_actions = 9
        network = AlphaNetwork(residual_depth, single_channel_zize, num_input_channels, num_possible_actions)
        network = network.double()
        prediction_network = TickTackToePredictionNetwork(network)

        # agentA = AlphaZeroAgent(TickTackToePredictionNetwork(), game_engine, num_simulations=100)
        # agentB = AlphaZeroAgent(TickTackToePredictionNetwork(), game_engine, num_simulations=100)
        # evaluation = Evaluation(game_engine, agentA, agentB)
        # scores[evaluation.play()] += 1

        self_play_engine = SelfPlay(prediction_network, game_engine, 1000)
        game_score, training_examples = self_play_engine.play()
        all_examples.extend(training_examples)
        scores[game_score] += 1

    print(f"scores:{scores}, number examples: {len(training_examples)}")
    print("Attempting to train")
    losses = network.train(all_examples, epochs=100, batch_size=32)
    print(f"losses: {losses}")



# board = TickTakToeBoard()
# board = board.make_move(1, 1, 1)
# print(board)
# print(board.game_over())
#
# board = board.make_move(1, 1, 2)
# print(board)
# print(board.game_over())
#
# board = board.make_move(1, 0, 0)
# print(board)
# print(board.game_over())
#
# board = board.make_move(2, 1, 0)
# print(board)
# print(board.game_over())
#
# board = board.make_move(1, 2, 0)
# print(board)
# print(board.game_over())
#
# board = board.make_move(2, 2, 1)
# print(board)
# print(board.game_over())
#
# board = board.make_move(2, 2, 2)
# print(board)
# print(board.game_over())
#
# board = board.make_move(1, 0, 1)
# print(board)
# print(board.game_over())
#
# board = board.make_move(2, 0, 2)
# print(board)
# print(board.game_over())
# print(board.get_game_score())

__main__()



