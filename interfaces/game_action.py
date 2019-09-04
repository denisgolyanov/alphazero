import abc


class GameAction(object, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __str__(self):
        """
        :return: a visual representation of the game action
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def __eq__(self, other):
        """
        :return: a visual representation of the game action
        """
        raise NotImplementedError()