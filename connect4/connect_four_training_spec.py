from connect4.connect_four_game_engine import ConnectFourGameEngine
from connect4.connect_four_prediction_network import ConnectFourPredictionNetwork
from interfaces.training_spec import TrainingSpecification

NUM_SIMULATIONS = 200
NUM_EVALUATE_GAMES = 40
NUM_RANDOM_GAMES = 40
RESIDUAL_DEPTH = 4
THRESHOLD = 1.05
NUM_EPOCHS = 10
NUM_INPUT_CHANNELS = 3
NUM_EPISODES = 100
NUM_GAMES_PER_EPISODE = 2
NUM_HISTORY_EPISODES = 5
NUM_ROWS = 6
NUM_COLS = 7


class ConnectFourTrainingSpecification(TrainingSpecification):

    def __init__(self):
        self._rows = NUM_ROWS
        self._cols = NUM_COLS

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
        return ConnectFourGameEngine(self._rows, self._cols)

    @property
    def num_input_channels(self):
        return NUM_INPUT_CHANNELS

    @property
    def single_channel_size(self):
        return self._rows * self._cols

    @property
    def total_possible_actions(self):
        return self._cols

    @property
    def training_examples_per_game(self):
        # factor 2 due to generated symmetries
        return self._rows * self._cols * 2

    def prediction_network(self, alpha_network):
        return ConnectFourPredictionNetwork(alpha_network, self._cols)

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
        return "connect_four"

    def training_augmentor(self):
        return ConnectFourTrainingAugmentor()

