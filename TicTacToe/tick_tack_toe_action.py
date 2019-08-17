from interfaces.game_action import GameAction


class TickTackToeAction(GameAction):
    def __init__(self, row, col):
        self.row = row
        self.col = col

    def __str__(self):
        return "[ACTION (%d, %d)]" % (self.row, self.col)

