import abc


class TrainingSpecification(object, metaclass=abc.ABCMeta):
    @property
    @abc.abstractmethod
    def num_simulations(self):
        pass

    @property
    @abc.abstractmethod
    def num_evaluation_games(self):
        pass

    @property
    @abc.abstractmethod
    def num_random_evaluation_games(self):
        pass

    @property
    @abc.abstractmethod
    def residual_depth(self):
        pass

    @property
    @abc.abstractmethod
    def improvement_threshold(self):
        pass

    @property
    @abc.abstractmethod
    def num_epochs(self):
        pass

    @abc.abstractmethod
    def game_engine(self):
        """
        :return: a GameEngine object
        """
        pass

    @property
    @abc.abstractmethod
    def num_input_channels(self):
        pass

    @property
    @abc.abstractmethod
    def single_channel_size(self):
        pass

    @property
    @abc.abstractmethod
    def total_possible_actions(self):
        pass

    @property
    @abc.abstractmethod
    def training_examples_per_game(self):
        pass

    @abc.abstractmethod
    def prediction_network(self, alpha_network):
        """

        :param alpha_network: a AlphaNetwork network object to wrap
        :return: a PredictionNetwork object corresponding to the game
        """
        pass

    @property
    @abc.abstractmethod
    def num_episodes(self):
        pass

    @property
    @abc.abstractmethod
    def num_games_per_episode(self):
        pass

    @property
    @abc.abstractmethod
    def num_history_episodes(self):
        pass

    @property
    @abc.abstractmethod
    def game_name(self):
        pass

    def __str__(self):
        return f""" {self.game_name}
        residual depth: {self.residual_depth}
        num simulations: {self.num_simulations}
        num episodes: {self.num_episodes}
        games per episodes: {self.num_games_per_episode}
        history episodes: {self.num_history_episodes}
        num epochs: {self.num_epochs}
        num evaluation games: {self.num_evaluation_games}
        num random evaluation games: {self.num_random_evaluation_games}
        improvement threshold: {self.improvement_threshold}
        """
