import logging
import numpy as np
from operator import attrgetter
from mcts.tree_node import TreeNode
from interfaces.game_state import GameState


class MCTSSimulator(object):
    def __init__(self, network, initial_state):
        """
        :param network: a PredictionNetwork implementation object
        :param initial_state: an initial GameState object
        """
        self.root = TreeNode(network, parent_node=None,
                             probability_estimation=None, action=None,
                             state=initial_state)

    def update_root_by_action(self, action):
        new_root = self.root.search_child(action=action)
        if new_root.state is None:
            new_root.eval_state()
        new_root.parent_node = None
        self.root = new_root

    def _single_simulation(self):
        node = self.root
        while not node.is_leaf() and not node.is_game_end():
            node = max(node.children, key=attrgetter('selection_score'))

        if node.is_game_end():
            value = node.state.get_game_score()
            assert value in [0, GameState.PLAYER_ONE, GameState.PLAYER_TWO]
            if value == 0:
                # tied game
                pass

            elif node.parent_node.state.get_player() == value:
                # player who made last move, won
                value = 1.0
            else:
                # player who made last move, lost
                value = -1.0

        else:
            # the value is the winning chance of the next player, thus the negation
            value = node.expand() * -1.0

        while not node.is_root():
            node.backwards_pass(value)
            node = node.parent_node
            value *= -1.0

        node.backwards_pass(value)

    def execute_simulations(self, num_simulations):
        for i in range(num_simulations):
            logging.debug(f"Simulation {i}")
            self._single_simulation()

    def compute_next_action_probabilities(self, competitive):
        """

        :return: A list of pairs <action, probability>, where probability is derived from the
                 MCTS tree, such that, probability = visit_count / total_visit_count

        :note:   the output will contain only the moves deemed legal by the game, probability
                 for discarded moves is implicitly zero.
        """

        if competitive:
            max_visit_count_node = max(self.root.children, key=attrgetter('visit_count'))

            def _child_node_probability(child_node):
                return 1.0 if child_node == max_visit_count_node else 0.0
        else:
            total_visit_count = sum([child_node.visit_count for child_node in self.root.children])

            def _child_node_probability(child_node):
                return child_node.visit_count / (1.0 * total_visit_count)

        probabilities = [(child_node.action, _child_node_probability(child_node)) for child_node in self.root.children]
        logging.info('Action probabilities: %s', str([prob for action, prob in probabilities]))
        logging.info('Children: %s', str([str(child_node) for child_node in self.root.children]))
        logging.info('Scores: %s', str([str(child_node.selection_score) for child_node in self.root.children]))

        return probabilities

