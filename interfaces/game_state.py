import abc


class GameState(object, metaclass=abc.ABCMeta):
    PLAYER_ONE = 1
    PLAYER_TWO = 2

    @abc.abstractmethod
    def do_action(self, action):
        """
        Execute given action on given state, and return the resulting game state.
        Action is not performed in-place, the resultant game state must allocate its own memory.

        :param action: the action to perform
        :rtype action: GameAction
        :return: a resultant GameState object
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def get_player(self):
        """
        :return: The current player, either PLAYER_ONE or PLAYER_TWO
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def get_game_score(self):
        """
        :return: Assuming game is over, this method either returns 0 in case of a tie, or the identity
        of the winning player, either PLAYER_ONE or PLAYER_TWO
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def game_over(self):
        """
        :return: True iff the state represents a game that has ended
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def all_possible_actions(self):
        """
        :return: a list of possible GameAction objects, assuming current game state
        if no moves are possible from current state, an empty list is returned.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def __str__(self):
        """
        :return: a visual representation of current state
        """
        raise NotImplementedError()
