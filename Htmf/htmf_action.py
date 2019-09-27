from interfaces.game_action import GameAction

class HtmfAction(GameAction):
    def __init__(self, start_coords, end_coords):
        self.start_coords = start_coords
        self.end_coords = end_coords

    def __str__(self):
        return "[ACTION {} -> {}]".format(self.start_coords, self.end_coords)

    def __eq__(self, other):
        return self.start_coords == other.start_coords and self.end_coords == other.end_coords

    def __repr__(self):
        return str(self)
