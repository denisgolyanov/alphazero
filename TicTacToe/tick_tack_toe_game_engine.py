from interfaces.game_engine import GameEngine
from TicTacToe.tick_tack_toe_state import TickTackToeState
from TicTacToe.tick_tack_toe_board import TickTakToeBoard


class TickTackToeGameEngine(GameEngine):
    def create_new_game(self):
        return TickTackToeState(TickTakToeBoard(), TickTackToeState.PLAYER_ONE)

