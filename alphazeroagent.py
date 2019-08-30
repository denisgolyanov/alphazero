from mcts.simulator import MCTSSimulator

import numpy as np


class AlphaZeroAgent(object):
    def __init__(self, prediction_network, game_engine, num_simulations):
        self.prediction_network = prediction_network
        self.game_engine = game_engine
        self.num_simulations = num_simulations
        self.simulator = self._create_new_simulator()

    def _create_new_simulator(self):
        game_state = self.game_engine.create_new_game()
        simulator = MCTSSimulator(self.prediction_network, game_state)
        self.simulator = simulator
        return simulator

    def update_simulator(self, action):
        self.simulator.update_root_by_action(action)

    def choose_action(self, fetch_probabilities=False):
        self.simulator.execute_simulations(num_simulations=self.num_simulations)
        action_probability_pairs = self.simulator.compute_next_action_probabilities()
        action = self._select_next_action(action_probability_pairs)
        self.update_simulator(action)

        if fetch_probabilities:
            return action, self.prediction_network.translate_to_action_probabilities_tensor(action_probability_pairs)

        return action

    @staticmethod
    def _select_next_action(action_probability_pairs):
        competitive = False  # TODO: extract to parameter

        assert not competitive, "Need to implement"
        # if competitive:
        #    return max(action_probability_pairs, key=lambda pair: child_node.visit_count)[0]

        return np.random.choice([pair[0] for pair in action_probability_pairs],
                                p=[pair[1] for pair in action_probability_pairs])
