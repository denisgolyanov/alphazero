import itertools

from interfaces.game_state import GameState


class TickTakToeBoard(object):
    def __init__(self):
        self._board = [[0, 0, 0],
                       [0, 0, 0],
                       [0, 0, 0]]

    def _is_board_full(self):
        return all([self._board[i][j] != 0 for i, j in itertools.product(range(3), range(3))])

    def _compute_all_triples(self):
        rows = [self._board[i] for i in range(3)]
        cols = [[self._board[i][j] for i in range(3)] for j in range(3)]
        all_triplets = rows + cols \
                       + [[self._board[0][0], self._board[1][1], self._board[2][2]]] \
                       + [[self._board[2][0], self._board[1][1], self._board[0][2]]]
        return all_triplets

    def game_over(self):
        if self._is_board_full():
            return True
        all_triples = self._compute_all_triples()
        return [1, 1, 1] in all_triples or [2, 2, 2] in all_triples

    def get_game_score(self):
        assert self.game_over()
        all_triples = self._compute_all_triples()
        if [1, 1, 1] in all_triples:
            return GameState.PLAYER_ONE
        elif [2, 2, 2] in all_triples:
            return GameState.PLAYER_TWO
        return 0

    def _clone(self):
        new_board = TickTakToeBoard()
        for i, j in itertools.product(range(3), range(3)):
            new_board._board[i][j] = self._board[i][j]
        return new_board

    def make_move(self, player, row, col):
        assert self._board[row][col] == 0
        assert player in [1, 2]
        new_board = self._clone()
        new_board._board[row][col] = player
        return new_board

    def __str__(self):
        result = ""
        for i in range(3):
            for j in range(3):
                val = self._board[i][j]
                if val == 1:
                    result += "X"
                elif val == 2:
                    result += "O"
                else:
                    result += "_"
                result += " "
            result += "\r\n"
        return result

    def __getitem__(self, item):
        i, j = item
        return self._board[i][j]

    def __setitem__(self, item, value):
        i, j = item
        self._board[i][j] = value

    def __eq__(self, other):
        return self._board == other._board