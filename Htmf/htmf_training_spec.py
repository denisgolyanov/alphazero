from Htmf.htmf_game_engine import HtmfGameEngine
from Htmf.htmf_prediction_network import HtmfPredictionNetwork
from interfaces.training_spec import TrainingSpecification

NUM_SIMULATIONS = 1000
NUM_EVALUATE_GAMES = 40
NUM_RANDOM_GAMES = 14
RESIDUAL_DEPTH = 6
THRESHOLD = 1.05
NUM_EPOCHS = 10
NUM_INPUT_CHANNELS = 4
NUM_EPISODES = 100
NUM_GAMES_PER_EPISODE = 50
NUM_HISTORY_EPISODES = 10

LENGTH = 5
PENGUINS_PER_PLAYER = 2


class HtmfTrainingSpecification(TrainingSpecification):

    def __init__(self):
        self._length = LENGTH
        self._penguins_per_player = PENGUINS_PER_PLAYER

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
        return HtmfGameEngine(self._length)

    @property
    def num_input_channels(self):
        return NUM_INPUT_CHANNELS

    @property
    def single_channel_size(self):
        return self._length ** 2

    @property
    def total_possible_actions(self):
        return (self._length ** 2) * 2

    @property
    def training_examples_per_game(self):
        return (self._length ** 2)

    def prediction_network(self, alpha_network):
        return HtmfPredictionNetwork(alpha_network, self._length, self._penguins_per_player)

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
        return "htmf"

    def training_augmentor(self):
        return None

    @property
    def temperature(self):
        return self._length ** 2 - self._length * 2
