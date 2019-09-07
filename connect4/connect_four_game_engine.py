from interfaces.game_engine import GameEngine
from connect4.connect_four_state import ConnectFourState
from connect4.connect_four_board import ConnectFourBoard


class ConnectFourGameEngine(GameEngine):
    def __init__(self, rows, cols):
        self._rows = rows
        self._cols = cols

    def create_new_game(self):
        return ConnectFourState(ConnectFourBoard(self._rows, self._cols), ConnectFourState.PLAYER_ONE)

