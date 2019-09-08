from interfaces.game_state import GameState
from Htmf.htmf_action import HtmfAction


class HtmfState(GameState):
    def __init__(self, board, current_player):
        self._board = board
        self._current_player = current_player

        assert current_player in [self.PLAYER_ONE, self.PLAYER_TWO]

    def _next_player(self):
        next_player = self.PLAYER_ONE if self._current_player == self.PLAYER_TWO else self.PLAYER_TWO

        if self._board.can_move(self._current_player):
            return next_player
        else:
            return self._current_player

    def do_action(self, action):
        assert isinstance(action, HtmfAction)
        new_board = self._board.make_move(action.start_coords, action.end_coords)

        return HtmfState(new_board, self._next_player())

    def get_player(self):
        return self._current_player

    def get_game_score(self):
        assert self.game_over()
        
        scores = self._board.get_game_score()
        p1score = scores[GameState.PLAYER_ONE]
        p2score = scores[GameState.PLAYER_TWO]
        
        if p1score > p2score:
            return GameState.PLAYER_ONE
        elif p1score < p2score:
            return GameState.PLAYER_TWO
        else:
            return 0

    def game_over(self):
        return self._board.game_over()

    def all_possible_actions(self):
        return (HtmfAction(start, end)
                for start, end in self._board.all_possible_moves_for_player(self._current_player))

    def __str__(self):
        return str(self._board) + "TURN: {0}\r\n".format(self._current_player)

    def convert_to_tensor(self):
        board_length = self._board.length
        tensor = torch.zeros([1, 4, board_length, board_length], dtype=torch.double)

        # player whose turn it is always appears first in tensor
        first_player = self._current_player
        second_player = self.PLAYER_ONE if first_player == self.PLAYER_TWO else self.PLAYER_TWO

        # one hot for "player has penguin here". penguins with smaller coordinates go first.
        # the sorting might not be necessary since the penguins' order shouldn't really change.
        for penguin in sorted(self.board.penguins, lambda p: p.bhex.coords()):
            if penguin.player == first_player:
                tensor[0, 0, penguin.bhex.x, penguin.bhex.y] = 1
            if penguin.player == second_player:
                tensor[0, 1, penguin.bhex.x, penguin.bhex.y] = 1

        # one hot for "player has taken this tile"
        for bhex in self.board.hexes.values():
            if bhex.player == first_player:
                tensor[0, 2, bhex.x, bhex.y] = 1
            elif bhex.player == second_player:
                tensor[0, 3, bhex.x, bhex.y] = 1

        return tensor
