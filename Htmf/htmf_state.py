from interfaces.game_state import GameState
from Htmf.htmf_action import HtmfAction


class HtmfState(GameState):
    def __init__(self, board, current_player):
        self._board = board
        self._current_player = current_player

        assert current_player in [self.PLAYER_ONE, self.PLAYER_TWO]

    def _next_player(self):
        if self.self._current_player == self.PLAYER_TWO and self._board.can_move(self.PLAYER_ONE):
            return self.PLAYER_ONE
        else:
            return self.PLAYER_TWO

    def do_action(self, action):
        assert isinstance(action, HtmfAction)
        new_board = self._board.make_move(action.start_coords, action.end_coords)

        return HtmfState(new_board, self._next_player())

    def get_player(self):
        return self._current_player

    def get_game_score(self):
        return self._board.get_game_score()

    def game_over(self):
        return self._board.game_over()

    def all_possible_actions(self):
        return (HtmfAction(start.coords(), end.coords())
                for start, end in self._board.all_possible_moves_for_player(self._current_player))

#    def __str__(self):
#        return str(self._board) + "TURN: {0}\r\n".format(self._current_player)

