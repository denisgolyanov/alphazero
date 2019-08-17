import abc


class GameAction(object, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __str__(self):
        """
        :return: a visual representation of the game action
        """
        raise NotImplementedError()

