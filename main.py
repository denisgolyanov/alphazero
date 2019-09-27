from Htmf.htmf_training_spec import HtmfTrainingSpecification
from train import train, compete_with_user, evaluatle_random_from_checkpoint
from utils import logger, CUDA
from TicTacToe.tick_tack_toe_training_spec import TickTackToeTrainingSpecification
from connect4.connect_four_training_spec import ConnectFourTrainingSpecification

logger.info(f"Using cuda: {CUDA}")


# TickTacToe entry points
compete_with_user(TickTackToeTrainingSpecification(), "tictactoe_best")
train(TickTackToeTrainingSpecification())
evaluatle_random_from_checkpoint(TickTackToeTrainingSpecification(), checkpoint="tictactoe_best")

# ConnectFour entry poitns
evaluatle_random_from_checkpoint(ConnectFourTrainingSpecification(), checkpoint="connect_four_best")
train(ConnectFourTrainingSpecification())
compete_with_user(ConnectFourTrainingSpecification(), "connect_four_best")

# HTMF entry points
# TODO: complete