from interfaces.game_action import GameAction


class ConnectFourAction(GameAction):
    def __init__(self, col):
        self.col = col

    def __str__(self):
        return "[ACTION (%d)]" % self.col

    def __eq__(self, other):
        return self.col == other.col

    def __repr__(self):
        return str(self)
