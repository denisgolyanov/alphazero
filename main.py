from Htmf.htmf_training_spec import HtmfTrainingSpecification
from train import train, compete_with_user
from utils import logger, CUDA
from TicTacToe.tick_tack_toe_training_spec import TickTackToeTrainingSpecification
from connect4.connect_four_training_spec import ConnectFourTrainingSpecification

logger.info(f"Using cuda: {CUDA}")

# train(TickTackToeTrainingSpecification())
#train(HtmfTrainingSpecification(), checkpoint="htmf2019-09-19T10_30_38_554978")
compete_with_user(HtmfTrainingSpecification(), "htmf2019-09-19T18_54_07_597763")


# Complete TickTacToe model
#compete_with_user("2019-09-06T12_10_29_397881")



