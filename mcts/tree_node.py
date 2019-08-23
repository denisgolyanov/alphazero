
"""
Represents a possible move in the MCTS search tree
Each move is a pair <action, probability>, where probability is the probability score
assigned by the neural network.
"""
class TreeNode(object):
    def __init__(self, network, parent_node, action, probability_estimation):

        self.network = network

        self.parent_node = parent_node

        # Total number tree node was visited during simulation
        self.visit_count = 0

        # The total action value, in this node and all children nodes
        self.total_action_value = 0

        # total_action_value / visit_count
        self.mean_action_value = 0

        self.action = action

        # P(s, a), where s is the parent state and a is the action
        # leading towards this state
        self.probability_estimation = probability_estimation

        self.children = None
        self.state = None

    def __str__(self):
        return "[V: {0}, Q: {1:.2f}, P: {2:.2f}, A:{3}]".format(self.visit_count, self.mean_action_value,
                                                                self.probability_estimation, self.action)

    def is_leaf(self):
        return self.children is None

    def is_root(self):
        return self.parent_node is None

    def is_game_end(self):
        """
        :return: True iff the expanded node represents a game end
        """
        return not self.is_leaf() and len(self.children) == 0

    @property
    def selection_score(self):
        """
        :return: The selection score for this state, action pair
                 A variant of the PUCT algorithm
        """
        C_PUCT = 0.2

        # The exploration score diminishes as the visit count increases
        exploration_score = \
            C_PUCT * self.probability_estimation * (self.parent_node.visit_count ** 0.5) / (1.0 + self.visit_count)
        return self.mean_action_value + exploration_score

    def backwards_pass(self, v):
        """
        Apply the backwards pass modifications for given node
        :param v: the value predicted by the network
        """

        self.visit_count += 1
        self.total_action_value += v
        self.mean_action_value = self.total_action_value / (1.0 * self.visit_count)

    def expand(self):
        """
        Expands current node in the simulation, by generating all the possible actions
        from current state, and fetching their probabilities using the network
        :return: the value prediction for state represented by the node
        """

        # apply action on parent state, to create a new state representation
        if self.is_root():
            assert self.state is not None
        else:
            assert self.state is None
            self.state = self.parent_node.state.do_action(self.action)

        # apply network on new state
        action_probability_pairs, value = self.network.predict(self.state)

        self.children = \
            [TreeNode(self.network, self, action, probability) for action, probability in action_probability_pairs]

        if self.state.game_over():
            assert len(self.children) == 0

        return value

    def search_child(self, action, state):
        assert action is not None or state is not None
        if action and not state:
            state = self.state.do_action(self.action)
        for child in self.children:
            #TODO: is this comparison legit?
            if child.state == state:
                return child


class RootNode(TreeNode):
    def __init__(self, network, initial_state):
        super().__init__(network, None, None, None)

        self.state = initial_state
