from train import train, compete_with_user
from utils import logger, CUDA
from TicTacToe.tick_tack_toe_training_spec import TickTackToeTrainingSpecification
from connect4.connect_four_training_spec import ConnectFourTrainingSpecification

logger.info(f"Using cuda: {CUDA}")

# train(TickTackToeTrainingSpecification())
train(ConnectFourTrainingSpecification())
#compete_with_user(ConnectFourTrainingSpecification(), "connect_four2019-09-06T14_47_05_079127")


# Complete TickTacToe model
#compete_with_user("2019-09-06T12_10_29_397881")



