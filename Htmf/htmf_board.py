from itertools import product, chain
from interfaces.game_state import GameState

class HexesNotInLineException(Exception):
    pass
class IllegalMoveException(Exception):
    pass

class HtmfBoard(object):
    def __init__(self, length, hexes=None, penguins=None, scores=None):
        assert (hexes is None and penguins is None and scores is None) or (hexes is not None and penguins is not None and scores is not None)

        # make new board
        if hexes is None:
            self.hexes = {(x, y) : Hex(x, y) for x, y in 
                            (coords for coords in product(range(length), repeat=2))}
            self.penguins = set()
            self.scores = {GameState.PLAYER_ONE: 0, GameState.PLAYER_TWO: 0}
            self.length = length

        # copy old board
        else:
            self.hexes = hexes
            self.penguins = penguins
            self.scores = scores
            self.length = length

    def get_hex(self, coords):
        return self.hexes[coords]

    def update_penguins_movability(self):
        for penguin in self.penguins:
            penguin.can_move = any(n.is_passable for n in self.neighbors(penguin.bhex))

    def place_penguin(self, coords, player):
        bhex = self.hexes[coords]

        if not bhex.is_passable:
            raise IllegalMoveException()

        self.penguins.add(Penguin(bhex, player))
        bhex.is_passable = False
        bhex.player = player
        self.scores[player] += bhex.value

        self.update_penguins_movability()


    @staticmethod
    def get_common_axis(coords1, coords2):
        for i in range(2):
            if coords1[i] == coords2[i]:
                return i

        if sum(coords1) == sum(coords2):
            return 2

        return None

    def hexes_from_to(self, start_hex, end_hex):
        assert start_hex is not end_hex
        
        coords1 = start_hex.coords()
        coords2 = end_hex.coords()
        line_hexes = set()

        common_axis = HtmfBoard.get_common_axis(coords1, coords2)
        if common_axis is None:
            raise HexesNotInLineException()

        if common_axis == 0:
            axis_directions = (0, (1 if coords1[1 - common_axis] < coords2[1 - common_axis] else -1))
        if common_axis == 1:
            axis_directions = ((1 if coords1[1 - common_axis] < coords2[1 - common_axis] else -1), 0)
        if common_axis == 2:
            axis_directions = (1, -1) if coords1[0] < coords2[0] else (-1, 1)

        cur_coords = coords1
        while cur_coords != coords2:
            cur_coords = tuple(map(sum, zip(cur_coords, axis_directions)))
            line_hexes.add(self.hexes[cur_coords])

        return line_hexes

    def is_passable_path(self, start_hex, end_hex):
        try:
            return (start_hex is not end_hex) and all(bhex.is_passable for bhex in self.hexes_from_to(start_hex, end_hex))
        except HexesNotInLineException:
            return False

    # we can optimize this if necessary
    def all_possible_moves_for_penguin(self, penguin):
        start_hex = penguin.bhex
        return ((start_hex.coords(), bhex.coords()) for bhex in self.hexes.values() if self.is_passable_path(start_hex, bhex))

    def all_possible_moves_for_player(self, player):
        return chain(*(self.all_possible_moves_for_penguin(penguin) 
                       for penguin in self.penguins
                       if penguin.player == player))

    def is_passable_path_coords(self, start_coords, end_coords):
        return self.is_passable_path(self.hexes[start_coords], self.hexes[end_coords])

    def neighbors(self, bhex):
        x, y = bhex.x, bhex.y
        coords = [(x - 1, y),
                  (x, y - 1),
                  (x + 1, y - 1),
                  (x + 1, y),
                  (x, y + 1),
                  (x - 1, y + 1)]

        return (self.hexes[c] for c in coords if c in self.hexes)

    def game_over(self):
        return not any(penguin.can_move for penguin in self.penguins)

    def get_game_score(self):
        return self.scores

    def _clone(self):
        new_hexes = {coords.coords() : coords.clone() for coords in self.hexes.values()}
        new_penguins = set(Penguin(new_hexes[p.bhex.coords()], p.player)
                           for p in self.penguins)

        return HtmfBoard(length=self.length, hexes=new_hexes, penguins=new_penguins, scores=self.scores.copy())

    def make_move(self, start_coords, end_coords):
        assert self.is_passable_path_coords(start_coords, end_coords)

        new_board = self._clone()
        penguin = [p for p in new_board.penguins if p.bhex.coords() == start_coords][0]
        target_hex = new_board.hexes[end_coords]

        penguin.bhex = target_hex
        target_hex.player = penguin.player
        target_hex.is_passable = False
        new_board.scores[penguin.player] += target_hex.value

        new_board.update_penguins_movability()

        return new_board

    def can_move(self, player):
        return any(penguin.can_move for penguin in self.penguins if penguin.player == player)

    def __str__(self):
        return '\n'.join(str(h) for h in self.hexes.values())


class Hex(object):
    def __init__(self, x, y, value=1, is_passable=True, player=None):
        self.x = x
        self.y = y
        self.value = value
        self.is_passable = is_passable
        self.player = player

    def clone(self):
        return Hex(self.x, self.y, self.value, self.is_passable, self.player)

    def __str__(self):
        return 'Hex(x={}, y={}, is_passable={}, player={})'.format(self.x,
                                                                   self.y,
                                                                   self.is_passable,
                                                                   self.player)

    def coords(self):
        return (self.x, self.y)

class Penguin(object):
    def __init__(self, bhex, player):
        self.bhex = bhex
        self.player = player
        self.can_move = True
