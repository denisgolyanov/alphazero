from TicTacToe.tick_tack_toe_game_engine import TickTackToeGameEngine
from TicTacToe.tick_tack_toe_prediction_network import TickTackToePredictionNetwork
from interfaces.training_spec import TrainingSpecification

NUM_SIMULATIONS = 200
NUM_EVALUATE_GAMES = 40
NUM_RANDOM_GAMES = 40
RESIDUAL_DEPTH = 2
THRESHOLD = 1.05
NUM_EPOCHS = 10
NUM_INPUT_CHANNELS = 3
SINGLE_CHANNEL_SIZE = 9
NUM_POSSIBLE_ACTIONS = 9
TRAINING_EXAMPLES_PER_GAME = 9
NUM_EPISODES = 100
NUM_GAMES_PER_EPISODE = 50
NUM_HISTORY_EPISODES = 5


class TickTackToeTrainingSpecification(TrainingSpecification):

    @property
    def num_simulations(self):
        return NUM_SIMULATIONS

    @property
    def num_evaluation_games(self):
        return NUM_EVALUATE_GAMES

    @property
    def num_random_evaluation_games(self):
        return NUM_RANDOM_GAMES

    @property
    def residual_depth(self):
        return RESIDUAL_DEPTH

    @property
    def improvement_threshold(self):
        return THRESHOLD

    @property
    def num_epochs(self):
        return NUM_EPOCHS

    def game_engine(self):
        return TickTackToeGameEngine()

    @property
    def num_input_channels(self):
        return NUM_INPUT_CHANNELS

    @property
    def single_channel_size(self):
        return SINGLE_CHANNEL_SIZE

    @property
    def total_possible_actions(self):
        return NUM_POSSIBLE_ACTIONS

    @property
    def training_examples_per_game(self):
        return TRAINING_EXAMPLES_PER_GAME

    def prediction_network(self, alpha_network):
        return TickTackToePredictionNetwork(alpha_network)

    @property
    def num_episodes(self):
        return NUM_EPISODES

    @property
    def num_games_per_episode(self):
        return NUM_GAMES_PER_EPISODE

    @property
    def num_history_episodes(self):
        return NUM_HISTORY_EPISODES

    @property
    def game_name(self):
        return "tick_tack_toe"

