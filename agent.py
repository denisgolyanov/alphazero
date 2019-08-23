from mcts.simulator import MCTSSimulator
from interfaces.game_engine import GameEngine

class Agent(object):

    def __init__(self, network, game_engine):
        self.network = network
        self.game_engine = game_engine
        self.simulator = self.create_new_simulator()

    def create_new_simulator(self, game_state=None):
        if game_state is None:
            game_state = self.game_engine.create_new_game()
        simulator = MCTSSimulator(self.network, game_state)
        self.simulator = simulator
        return simulator

    def update_simulator(self, action=None, state=None):
        assert action or state
        if action:
            new_root = self.simulator.root.search_child(action=action)
        else:
            new_root = self.simulator.root.search_child(state=state)
        self.simulator.update_root(new_root)


    def choose_action(self):
        action = self.simulator.compute_next_action()
        self.update_simulator(action=action)
        return action