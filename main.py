from time import time

from self_play import SelfPlay
from TicTacToe.tick_tack_toe_prediction_network import TickTackToePredictionNetwork
from TicTacToe.tick_tack_toe_game_engine import TickTackToeGameEngine
from interfaces.game_state import GameState

from Htmf.htmf_prediction_network import HtmfPredictionNetwork
from Htmf.htmf_game_engine import HtmfGameEngine

import logging

logging.basicConfig(level=logging.WARN)
logging.info("Hello world")


def __main__():
    times = []

#    scores = {0 : 0, GameState.PLAYER_ONE: 0, GameState.PLAYER_TWO: 0}
#    times.append(time())
#    for i in range(10):
#        self_play_engine = SelfPlay(TickTackToePredictionNetwork(), TickTackToeGameEngine())
#        scores[self_play_engine.play()] += 1
#    times.append(time())
#    print(scores)

    scores = {0 : 0, GameState.PLAYER_ONE: 0, GameState.PLAYER_TWO: 0}
    times.append(time())
    for i in range(10):
        self_play_engine = SelfPlay(HtmfPredictionNetwork(), HtmfGameEngine())
        scores[self_play_engine.play()] += 1
    times.append(time())
    print(scores)

    print(times)
    print(times[1] - times[0])
#    print(times[3] - times[2])


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

if __name__ == '__main__':
    __main__()
