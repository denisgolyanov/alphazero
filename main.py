from alphazeroagent import AlphaZeroAgent
from evaluation import Evaluation
from randomagent import RandomAgent
from self_play import SelfPlay
from TicTacToe.tick_tack_toe_prediction_network import TickTackToePredictionNetwork
from TicTacToe.tick_tack_toe_game_engine import TickTackToeGameEngine
from interfaces.game_state import GameState
from alpha_network.alpha_network import AlphaNetwork

import logging
import copy

logging.basicConfig(level=logging.WARN)
logging.info("Hello world")


NUM_SIMULATIONS = 200
NUM_EVALUATE_GAMES = 1
NUM_RANDOM_GAMES = 5
RESIDUAL_DEPTH = 2
THRESHOLD = 1.05
NUM_EPOCHS = 10


def __main__():

    # specifications for tic-tack-toe
    game_engine = TickTackToeGameEngine()
    single_channel_zize = 9
    num_input_channels = 1
    num_possible_actions = 9
    training_examples_per_game = 9

    previous_network = AlphaNetwork(RESIDUAL_DEPTH, single_channel_zize, num_input_channels, num_possible_actions)
    previous_network = previous_network.double()

    num_episodes = 100
    num_games_per_episode = 1
    num_history_episodes = 5
    num_games_history = num_games_per_episode * num_history_episodes

    num_examples_history = num_games_history * training_examples_per_game
    all_examples = []

    for episode in range(num_episodes):
        print(f"episode: {episode}")

        current_network = copy.deepcopy(previous_network)

        for game in range(num_games_per_episode):
            self_play_engine = SelfPlay(TickTackToePredictionNetwork(current_network), game_engine, NUM_SIMULATIONS)
            game_score, training_examples = self_play_engine.play()
            all_examples.extend(training_examples)

        all_examples = list(reversed(list(reversed(all_examples))[:num_examples_history]))
        losses = current_network.train(all_examples, epochs=NUM_EPOCHS, batch_size=32)
        print(f"all_examples: {len(all_examples)}")

        if pit(game_engine, previous_network, current_network):
            print("Saving checkpoint")
            previous_network = current_network
            previous_network.save_checkpoint()
        else:
            # retain examples from previous episode
            pass

        evaluate_competitive(game_engine, previous_network, current_network)
        evaluate_random(game_engine, current_network)


def evaluate_competitive(game_engine, previous_network, current_network):
    previous_prediction_network = TickTackToePredictionNetwork(previous_network)
    current_prediction_network = TickTackToePredictionNetwork(current_network)

    agent_a = AlphaZeroAgent(previous_prediction_network, game_engine, num_simulations=NUM_SIMULATIONS)
    agent_b = AlphaZeroAgent(current_prediction_network, game_engine, num_simulations=NUM_SIMULATIONS)

    game_score = Evaluation(game_engine, agent_a, agent_b).play(competitive=True)
    print(f"Competitive evalulation score, new agent is 2nd player: {game_score}")

    agent_a = AlphaZeroAgent(previous_prediction_network, game_engine, num_simulations=NUM_SIMULATIONS)
    agent_b = AlphaZeroAgent(current_prediction_network, game_engine, num_simulations=NUM_SIMULATIONS)

    game_score = Evaluation(game_engine, agent_b, agent_a).play(competitive=True)
    print(f"Competitive evalulation score, new agent is 1st player: {game_score}")

def evaluate_random(game_engine, current_network):
    current_prediction_network = TickTackToePredictionNetwork(current_network)

    scores = {0 : 0, GameState.PLAYER_ONE: 0, GameState.PLAYER_TWO: 0}
    logging.info("eval random")

    for i in range(NUM_RANDOM_GAMES // 2):
        logging.warning(f"Eval random round {i}")

        agent_a = RandomAgent()
        agent_b = AlphaZeroAgent(current_prediction_network, game_engine, num_simulations=NUM_SIMULATIONS)
        evaluation = Evaluation(game_engine, agent_a, agent_b)

        scores[evaluation.play()] += 1

    for i in range(NUM_RANDOM_GAMES // 2):
        logging.warning(f"Eval random round {NUM_RANDOM_GAMES // 2 + i}")

        agent_a = AlphaZeroAgent(current_prediction_network, game_engine, num_simulations=NUM_SIMULATIONS)
        agent_b = RandomAgent()
        evaluation = Evaluation(game_engine, agent_a, agent_b)

        game_score = evaluation.play()
        if game_score == GameState.PLAYER_TWO:
            game_score = GameState.PLAYER_ONE
        else:
            game_score = GameState.PLAYER_TWO

        scores[game_score] += 1

    logging.warning(f"Eval scores vs random agent {scores}")

def pit(game_engine, previous_network, current_network):
    print("Evaluation has begun")
    previous_prediction_network = TickTackToePredictionNetwork(previous_network)
    current_prediction_network = TickTackToePredictionNetwork(current_network)

    scores = {0 : 0, GameState.PLAYER_ONE: 0, GameState.PLAYER_TWO: 0}

    for i in range(NUM_EVALUATE_GAMES // 2):
        logging.warning(f"Eval round {i}")

        agent_a = AlphaZeroAgent(previous_prediction_network, game_engine, num_simulations=NUM_SIMULATIONS)
        agent_b = AlphaZeroAgent(current_prediction_network, game_engine, num_simulations=NUM_SIMULATIONS)
        evaluation = Evaluation(game_engine, agent_a, agent_b)

        scores[evaluation.play()] += 1

    for i in range(NUM_EVALUATE_GAMES // 2):
        logging.warning(f"Eval round {NUM_EVALUATE_GAMES // 2 + i}")

        agent_a = AlphaZeroAgent(previous_prediction_network, game_engine, num_simulations=NUM_SIMULATIONS)
        agent_b = AlphaZeroAgent(current_prediction_network, game_engine, num_simulations=NUM_SIMULATIONS)
        evaluation = Evaluation(game_engine, agent_b, agent_a)

        game_score = evaluation.play()
        if game_score == GameState.PLAYER_TWO:
            game_score = GameState.PLAYER_ONE
        else:
            game_score = GameState.PLAYER_TWO

        scores[game_score] += 1

    print(f"Eval scores {scores}")
    if not scores[GameState.PLAYER_ONE] or scores[GameState.PLAYER_TWO] / (scores[GameState.PLAYER_ONE]) > THRESHOLD:
        print("Model has improved")
        return True
    else:
        print(f"No improvement")
        return False

__main__()



