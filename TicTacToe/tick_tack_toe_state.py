from TicTacToe.tick_tack_toe_board import TickTakToeBoard
from interfaces.game_state import GameState
from TicTacToe.tick_tack_toe_action import TickTackToeAction

import itertools
import torch

class TickTackToeState(GameState):
    def __init__(self, board, current_player):
        self._board = board
        self._current_player = current_player

        assert current_player in [self.PLAYER_ONE, self.PLAYER_TWO]

    def _next_player(self):
        return self.PLAYER_ONE if self._current_player == self.PLAYER_TWO else self.PLAYER_TWO

    def do_action(self, action):
        assert isinstance(action, TickTackToeAction)
        new_board = self._board.make_move(self._current_player, action.row, action.col)

        return TickTackToeState(new_board, self._next_player())

    def get_player(self):
        return self._current_player

    def get_game_score(self):
        return self._board.get_game_score()

    def game_over(self):
        return self._board.game_over()

    def all_possible_actions(self):
        actions = []
        if self.game_over():
            return actions

        for i in range(3):
            for j in range(3):
                if self._board[i, j] == 0:
                    actions.append(TickTackToeAction(i, j))
        return actions

    def __str__(self):
        return str(self._board) + "TURN: {0}\r\n".format(self._current_player)

    def __eq__(self, other):
        return self._board == other._board \
        and self._current_player == other._current_player

    def convert_to_tensor(self):
        tensor = torch.empty([1, 1, 3, 3], dtype=torch.double)
        for i in range(3):
            for j in range(3):
                tensor[0, 0, i, j] = self._board[i, j]

        return tensor

    @staticmethod
    def convert_from_tensor(tensor):
        board = TickTakToeBoard()
        for i in range(3):
            for j in range(3):
                board[i, j] = tensor[3*i + j]

        num_xes = sum([board[i, j] == 1 for i, j in itertools.product(range(3), range(3))])
        num_os = sum([board[i, j] == 2 for i, j in itertools.product(range(3), range(3))])

        turn = TickTackToeState.PLAYER_ONE if num_os == num_xes else TickTackToeState.PLAYER_TWO

        return TickTackToeState(board, turn)
