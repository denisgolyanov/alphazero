import abc


class GameEngine(object, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def create_new_game(self):
        """
        :return: an initial GameState object, representing the initial state of a game
        """
        raise NotImplementedError()

