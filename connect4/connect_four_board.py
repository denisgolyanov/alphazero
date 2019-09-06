import itertools


class ConnectFourBoard(object):
    def __init__(self, rows, cols):
        self._rows = rows
        self._cols = cols
        self._board = [[0 for j in range(cols)] for i in range(rows)]
        self._winning_player = 0

    def _is_board_full(self):
        return all([self._board[i][j] != 0 for i, j in itertools.product(range(self._rows), range(self._cols))])

    def _is_winning_move(self, player, row, col):

        winning_four = [player] * 4

        col_index_start = max(0, col - 3)
        col_index_end   = min(self._cols - 4, col)
        row_index_start = max(0, row - 3)
        row_index_end   = min(self._rows - 4, row)

        for col_index in range(col_index_start, col_index_end + 1):
            if winning_four == self._board[row][col_index: col_index + 4]:
                return True

        for row_index in range(row_index_start, row_index_end + 1):
            if winning_four == [self._board[i][col] for i in range(row_index, row_index + 4)]:
                return True

        decreasing_diagonal_start_offset = min(3, row, col)
        decreasing_diagonal_end_offset   = 3 - min(self._cols - col - 1, self._rows - row - 1, 3)

        offset = decreasing_diagonal_start_offset
        while decreasing_diagonal_end_offset <= offset:
            if winning_four == [self._board[row - offset + i][col - offset + i] for i in range(4)]:
                return True
            offset -= 1

        increasing_diagonal_start_offset = min(3, col, self._rows - row - 1)
        increasing_diagonal_end_offset   = 3 - min(3, row, self._cols - col - 1)

        offset = increasing_diagonal_start_offset
        while increasing_diagonal_end_offset <= offset:
            if winning_four == [self._board[row + offset - i][col - offset + i] for i in range(4)]:
                return True
            offset -= 1

        return False

    def game_over(self):
        return self._winning_player != 0 or self._is_board_full()

    def get_game_score(self):
        assert self.game_over()
        return self._winning_player

    def _clone(self):
        new_board = ConnectFourBoard(self._rows, self._cols)
        for i, j in itertools.product(range(self._rows), range(self._cols)):
            new_board._board[i][j] = self._board[i][j]
        return new_board

    def make_move(self, player, col):
        row = 0
        while row < self._rows and self._board[row][col] == 0:
            row += 1

        row -= 1

        assert self._board[row][col] == 0
        assert player in [1, 2]
        new_board = self._clone()
        new_board._board[row][col] = player

        if new_board._is_winning_move(player, row, col):
            new_board._winning_player = player

        return new_board

    @property
    def rows(self):
        return self._rows

    @property
    def cols(self):
        return self._cols

    def __str__(self):
        result = ""
        for i in range(self._rows):
            for j in range(self._cols):
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