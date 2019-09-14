from train import train, compete_with_user
from utils import logger, CUDA
from TicTacToe.tick_tack_toe_training_spec import TickTackToeTrainingSpecification
from connect4.connect_four_training_spec import ConnectFourTrainingSpecification

logger.info(f"Using cuda: {CUDA}")

# train(TickTackToeTrainingSpecification())
#train(ConnectFourTrainingSpecification(), checkpoint="connect_four2019-09-14T11_31_25_451736")
compete_with_user(ConnectFourTrainingSpecification(), "connect_four2019-09-14T11_31_25_451736")


# Complete TickTacToe model
#compete_with_user("2019-09-06T12_10_29_397881")



