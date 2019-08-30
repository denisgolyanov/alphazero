import abc


class PredictionNetwork(object, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def predict(self, state):
        """
        :param state: The game state object to predict for
        :rtype state: GameState
        :return: a pair (action_probability_pairs, value)
                 where action_probability_pairs is a list of (action, probability) pairs predicted by the network
                 the value is the probability that the current player will win the game, predicted by the network
                 each action in the return list must be a valid action, of type GameAction

        :note: if game has ended and no possible future actions remain, an empty list must be returned.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def translate_to_action_probabilities_tensor(self, action_mcts_probability_pairs):
        """

        :param action_mcts_probability_pairs: a list of pairs of form <action, mcts_probability>
                                              derived from the mcts simulation tree.
        :note: missing actions are to be interpreted as illegal from current state and with zero probability
        :return: a tensor representing the policy  network output probabilities
        """
        raise NotImplementedError()

