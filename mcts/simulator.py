import logging
import numpy as np
from operator import attrgetter
from mcts.tree_node import RootNode
from interfaces.game_state import GameState


class MCTSSimulator(object):
    def __init__(self, network, initial_state):
        """

        :param network: a PredictionNetwork implementation object
        :param initial_state: an initial GameState object
        """
        self.root = RootNode(network, initial_state)

    def update_root(self, root):
        self.root = root

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

    def _compute_action_probabilities(self, temperature):
        def _child_node_probability(child_node):
            return child_node.visit_count ** temperature / \
                (sum([child_node.visit_count ** temperature for child_node in self.root.children]))

        return [(child_node.action, _child_node_probability(child_node)) for child_node in self.root.children]

    def _compute_next_action_competitive(self):
        best_child_node = max(self.root.children, key=lambda child_node: child_node.visit_count)
        logging.info('Children: %s', str([str(child_node) for child_node in self.root.children]))
        logging.info('Scores: %s', str([str(child_node.selection_score) for child_node in self.root.children]))

        return best_child_node.action

    def _compute_next_action_stochastic(self):

        total_visit_count = sum([child_node.visit_count for child_node in self.root.children])

        def _child_node_probability(child_node):
            return child_node.visit_count / (1.0 * total_visit_count)

        probabilities = [_child_node_probability(child_node) for child_node in self.root.children]
        logging.info('Move probabilities: %s', str(probabilities))
        logging.info('Children: %s', str([str(child_node) for child_node in self.root.children]))
        logging.info('Scores: %s', str([str(child_node.selection_score) for child_node in self.root.children]))

        return np.random.choice(self.root.children, p=probabilities).action

    def compute_next_action(self, num_simulations=1600):
        for i in range(num_simulations):
            logging.debug(f"Simulation {i}")
            self._single_simulation()

        competitive = True  # TODO: extract to parameter

        if competitive:
            return self._compute_next_action_competitive()
        else:
            return self._compute_next_action_stochastic()
