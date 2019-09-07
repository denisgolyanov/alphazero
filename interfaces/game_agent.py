import abc


class GameAgent(object, metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def notify_of_action(self, action):
        """
        Notifies the agent of the action of a rival agent
        The agent should update its inner state according to the action.

        :param action: an action commited by a rival agent
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def choose_action(self, competitive, game_state=None, fetch_probabilities=False):
        """
        Selects the next action according to the agent strategy

        :param competitive: Whether to act competitively
        :param game_state:  The current game state [Not used for alpha zero agents]
        :param fetch_probabilities: Whether to return the action probabilities tensor
                                    Used only for alpha zero agents during self play training
        :return: The selected action
                 if fetch_probabilities, returns <selection_action, action_probabilities_tensor>
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def restart_game(self):
        """
        Prepares the agent for a new game
        """
        raise NotImplementedError()

