from interfaces.game_state import GameState
from connect4.connect_four_action import ConnectFourAction

import torch


class ConnectFourState(GameState):
    def __init__(self, board, current_player):
        self._board = board
        self._current_player = current_player

        assert current_player in [self.PLAYER_ONE, self.PLAYER_TWO]

    def _next_player(self):
        return self.PLAYER_ONE if self._current_player == self.PLAYER_TWO else self.PLAYER_TWO

    def do_action(self, action):
        assert isinstance(action, ConnectFourAction)
        new_board = self._board.make_move(self._current_player, action.col)

        return ConnectFourState(new_board, self._next_player())

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

        for j in range(self._board.cols):
            if self._board[0, j] == 0:
                actions.append(ConnectFourAction(j))
        return actions

    def __str__(self):
        return str(self._board) + "TURN: {0}\r\n".format(self._current_player)

    def __eq__(self, other):
        return self._board == other._board \
        and self._current_player == other._current_player

    def convert_to_tensor(self):
        tensor = torch.zeros([1, 3, self._board.rows, self._board.cols], dtype=torch.double)
        turn_value = 0 if self.get_player() == GameState.PLAYER_ONE else 1

        for i in range(self._board.rows):
            for j in range(self._board.cols):
                tensor[0, 0, i, j] = turn_value

        for i in range(self._board.rows):
            for j in range(self._board.cols):
                tensor[0, self._board[i, j], i, j] = 1

        return tensor

