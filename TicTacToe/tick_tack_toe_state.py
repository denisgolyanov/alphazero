from interfaces.game_state import GameState
from TicTacToe.tick_tack_toe_action import TickTackToeAction


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
