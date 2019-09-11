from interfaces.game_engine import GameEngine
from Htmf.htmf_state import HtmfState
from Htmf.htmf_board import HtmfBoard


class HtmfGameEngine(GameEngine):
    def create_new_game(self):
        board = HtmfBoard(length=5)
        board.place_penguin((1, 1), HtmfState.PLAYER_ONE)
        board.place_penguin((3, 3), HtmfState.PLAYER_ONE)
        board.place_penguin((3, 1), HtmfState.PLAYER_TWO)
        board.place_penguin((1, 3), HtmfState.PLAYER_TWO)

        return HtmfState(board, HtmfState.PLAYER_ONE)
