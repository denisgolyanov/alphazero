from alphazeroagent import AlphaZeroAgent
from evaluation import Evaluation
from self_play import SelfPlay
from TicTacToe.tick_tack_toe_prediction_network import TickTackToePredictionNetwork
from TicTacToe.tick_tack_toe_game_engine import TickTackToeGameEngine
from interfaces.game_state import GameState

import logging

logging.basicConfig(level=logging.WARN)
logging.info("Hello world")


def __main__():

    scores = {0 : 0, GameState.PLAYER_ONE: 0, GameState.PLAYER_TWO: 0}

    for i in range(10):
        game_engine = TickTackToeGameEngine()

        # agentA = AlphaZeroAgent(TickTackToePredictionNetwork(), game_engine, num_simulations=100)
        # agentB = AlphaZeroAgent(TickTackToePredictionNetwork(), game_engine, num_simulations=100)
        # evaluation = Evaluation(game_engine, agentA, agentB)
        # scores[evaluation.play()] += 1

        self_play_engine = SelfPlay(TickTackToePredictionNetwork(), game_engine)
        game_score, training_examples = self_play_engine.play()
        scores[game_score] += 1

        print(f"examples: {training_examples}")

    print(scores)



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



