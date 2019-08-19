from interfaces.game_action import GameAction

class HtmfAction(GameAction):
    def __init__(self, start_coords, end_coords):
        self.start_coords = start_coords
        self.end_coords = end_coords

    def __str__(self):
        return "[ACTION {} -> {}]".format(start_coords, end_coords)