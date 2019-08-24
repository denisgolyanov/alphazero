from mcts.simulator import MCTSSimulator
from interfaces.game_engine import GameEngine

class AlphaZeroAgent(object):

    def __init__(self, network, game_engine, num_simulations):
        self.network = network
        self.game_engine = game_engine
        self.num_simulations = num_simulations
        self.simulator = self.create_new_simulator()

    def create_new_simulator(self, game_state=None):
        if game_state is None:
            game_state = self.game_engine.create_new_game()
        simulator = MCTSSimulator(self.network, game_state)
        self.simulator = simulator
        return simulator

    def update_simulator(self, action):
        current_root = self.simulator.root
        new_root = current_root.search_child(action=action)
        self.simulator.update_root(new_root)


    def choose_action(self):
        action = self.simulator.compute_next_action(num_simulations=self.num_simulations)
        self.update_simulator(action=action)
        return action