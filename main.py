import logging

from alphazeroagent import AlphaZeroAgent
from evaluation import Evaluation
from randomagent import RandomAgent
from self_play import SelfPlay
from TicTacToe.tick_tack_toe_prediction_network import TickTackToePredictionNetwork
from TicTacToe.tick_tack_toe_game_engine import TickTackToeGameEngine
from interfaces.game_state import GameState
from alpha_network.alpha_network import AlphaNetwork
from utils import logger
import copy


NUM_SIMULATIONS = 200
NUM_EVALUATE_GAMES = 100  # must be even
NUM_RANDOM_GAMES = 100 # must be even
RESIDUAL_DEPTH = 2
THRESHOLD = 1.05
NUM_EPOCHS = 10


def __main__():

    # specifications for tic-tack-toe
    game_engine = TickTackToeGameEngine()
    single_channel_size = 9
    num_input_channels = 1
    num_possible_actions = 9
    training_examples_per_game = 9

    previous_network = AlphaNetwork(RESIDUAL_DEPTH, single_channel_size, num_input_channels, num_possible_actions)
    previous_network = previous_network.double()

    num_episodes = 100
    num_games_per_episode = 50
    num_history_episodes = 5
    num_games_history = num_games_per_episode * num_history_episodes

    num_examples_history = num_games_history * training_examples_per_game
    all_examples = []
    logger.info("Starting")
    for episode in range(num_episodes):
        logger.info(f"episode: {episode}")

        current_network = copy.deepcopy(previous_network)

        for game in range(num_games_per_episode):
            logger.debug(f"Episode {episode} - Self-Playing game number {game}")
            self_play_engine = SelfPlay(TickTackToePredictionNetwork(current_network), game_engine, NUM_SIMULATIONS)
            game_score, training_examples = self_play_engine.play()
            all_examples.extend(training_examples)

        all_examples = list(reversed(list(reversed(all_examples))[:num_examples_history]))
        losses = current_network.train(all_examples, epochs=NUM_EPOCHS, batch_size=32)
        logger.info(f"current size of all_examples is {len(all_examples)}")

        if evaluate_vs_previous(game_engine, previous_network, current_network):
            logger.info("Saving checkpoint")
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

    evaluation = Evaluation(game_engine, agent_a, agent_b, competitive=True)
    game_score = evaluation.play_n_games(2) # 1 game for each side, because they're deterministic
    logger.info(f"Competitive evalulation score, new agent is 1st player: {game_score}")

def evaluate_random(game_engine, current_network):
    logger.info("evaluating against random agent...")
    agent_a = RandomAgent()
    current_prediction_network = TickTackToePredictionNetwork(current_network)
    agent_b = AlphaZeroAgent(current_prediction_network, game_engine, num_simulations=NUM_SIMULATIONS)
    evaluation = Evaluation(game_engine, agent_a, agent_b)
    scores = evaluation.play_n_games(NUM_RANDOM_GAMES)

    logger.info(f"Eval scores vs random agent {scores}")

def evaluate_vs_previous(game_engine, previous_network, current_network):
    logger.info("Evaluation has begun")
    previous_prediction_network = TickTackToePredictionNetwork(previous_network)
    current_prediction_network = TickTackToePredictionNetwork(current_network)

    agent_a = AlphaZeroAgent(previous_prediction_network, game_engine, num_simulations=NUM_SIMULATIONS)
    agent_b = AlphaZeroAgent(current_prediction_network, game_engine, num_simulations=NUM_SIMULATIONS)
    evaluation = Evaluation(game_engine, agent_a, agent_b)
    scores = evaluation.play_n_games(NUM_EVALUATE_GAMES)

    logger.info(f"Eval scores {scores}")
    if not scores[GameState.PLAYER_ONE] or scores[GameState.PLAYER_TWO] / (scores[GameState.PLAYER_ONE]) > THRESHOLD:
        logger.info("Model has improved")
        return True
    else:
        logger.info(f"No improvement")
        return False


def compete_with_user(checkpoint_name):
    single_channel_zize = 9
    num_input_channels = 1
    num_possible_actions = 9

    game_engine = TickTackToeGameEngine()

    network = AlphaNetwork(RESIDUAL_DEPTH, single_channel_zize, num_input_channels, num_possible_actions)
    network = network.double()
    network.load_checkpoint(checkpoint_name)

    agent_a = AlphaZeroAgent(TickTackToePredictionNetwork(network), game_engine, num_simulations=NUM_SIMULATIONS)
    from useragent import UserAgent
    print(f"Result: {Evaluation(game_engine, agent_a, UserAgent()).play(True)}")

    agent_a = AlphaZeroAgent(TickTackToePredictionNetwork(network), game_engine, num_simulations=NUM_SIMULATIONS)
    print(f"Result: {Evaluation(game_engine, UserAgent(), agent_a).play(True)}")



#compete_with_user("2019-08-31T10_14_38_027338")
__main__()



