import copy
import pickle

from interfaces.training_spec import TrainingSpecification
from alpha_network.alpha_network import AlphaNetwork
from alpha_zero_agent import AlphaZeroAgent
from evaluation import Evaluation
from interfaces.game_state import GameState
from random_agent import RandomAgent
from self_play import SelfPlay
from utils import logger, CUDA
from user_agent import UserAgent


def train(train_specification, checkpoint=None):
    assert isinstance(train_specification, TrainingSpecification)
    logger.info(f"{train_specification}")

    previous_network = AlphaNetwork(train_specification.residual_depth,
                                    train_specification.single_channel_size,
                                    train_specification.num_input_channels,
                                    train_specification.total_possible_actions)
    previous_network = previous_network.double()

    all_examples = []
    if checkpoint:
        all_examples = previous_network.load_checkpoint(checkpoint)

    num_games_history = train_specification.num_games_per_episode * train_specification.num_history_episodes
    num_examples_history = num_games_history * train_specification.training_examples_per_game

    logger.info("Starting")
    for episode in range(train_specification.num_episodes):
        logger.info(f"episode: {episode}/{train_specification.num_episodes}")

        current_network = copy.deepcopy(previous_network)

        for game in range(train_specification.num_games_per_episode):
            logger.debug(f"Episode {episode} - Self-Playing game number {game}/{train_specification.num_games_per_episode}")
            self_play_engine = SelfPlay(train_specification.prediction_network(current_network),
                                        train_specification.game_engine(),
                                        train_specification.num_simulations,
                                        train_specification.training_augmentor(),
                                        train_specification.temperature)

            game_score, training_examples = self_play_engine.play()
            all_examples.extend(training_examples)

        all_examples = list(reversed(list(reversed(all_examples))[:num_examples_history]))
        logger.info(f"current size of all_examples is {len(all_examples)}")
        if CUDA:
            logger.info("will use cuda during training")
            current_network = current_network.cuda()

        losses = current_network.train(all_examples, epochs=train_specification.num_epochs, batch_size=64)
        current_network = current_network.cpu()

        if evaluate_vs_previous(train_specification, previous_network, current_network):
            logger.info("Saving checkpoint")
            previous_network = current_network
            previous_network.save_checkpoint(train_specification.game_name, all_examples)
            evaluate_competitive(train_specification, previous_network, current_network)
            evaluate_random(train_specification, current_network)
        else:
            # retain examples from previous episode, but store checkpoint regardless
            current_network.save_checkpoint(train_specification.game_name, all_examples)



def evaluate_competitive(train_spec, previous_network, current_network):
    previous_prediction_network = train_spec.prediction_network(previous_network)
    current_prediction_network = train_spec.prediction_network(current_network)

    agent_a = AlphaZeroAgent(previous_prediction_network, train_spec.game_engine(),
                             num_simulations=train_spec.num_simulations)
    agent_b = AlphaZeroAgent(current_prediction_network, train_spec.game_engine(),
                             num_simulations=train_spec.num_simulations)

    evaluation = Evaluation(train_spec.game_engine(), agent_a, agent_b, competitive=True)
    game_score = evaluation.play_n_games(2)  # 1 game for each side, because they're deterministic
    logger.info(f"Competitive evaluation score, new agent is 2st player: {game_score}")


def evaluate_random(train_spec, current_network):
    logger.info("evaluating against random agent...")
    agent_a = RandomAgent()
    current_prediction_network = train_spec.prediction_network(current_network)
    agent_b = AlphaZeroAgent(current_prediction_network, train_spec.game_engine(),
                             num_simulations=train_spec.num_simulations)
    evaluation = Evaluation(train_spec.game_engine(), agent_a, agent_b, competitive=True)
    scores = evaluation.play_n_games(train_spec.num_random_evaluation_games)

    logger.info(f"Eval scores vs random agent {scores}")


def evaluate_vs_previous(train_spec, previous_network, current_network):
    logger.info("Evaluation has begun")
    previous_prediction_network = train_spec.prediction_network(previous_network)
    current_prediction_network = train_spec.prediction_network(current_network)

    agent_a = AlphaZeroAgent(previous_prediction_network, train_spec.game_engine(),
                             num_simulations=train_spec.num_simulations)
    agent_b = AlphaZeroAgent(current_prediction_network,  train_spec.game_engine(),
                             num_simulations=train_spec.num_simulations)
    evaluation = Evaluation(train_spec.game_engine(), agent_a, agent_b)
    scores = evaluation.play_n_games(train_spec.num_evaluation_games)

    logger.info(f"Eval scores {scores}")
    if not scores[GameState.PLAYER_ONE]\
       or scores[GameState.PLAYER_TWO] / (scores[GameState.PLAYER_ONE]) > train_spec.improvement_threshold:
        logger.info("Model has improved")
        return True
    else:
        logger.info(f"No improvement")
        return False


def compete_with_user(train_spec, checkpoint_name):

    network = AlphaNetwork(train_spec.residual_depth,
                           train_spec.single_channel_size,
                           train_spec.num_input_channels,
                           train_spec.total_possible_actions)
    network = network.double()
    if CUDA:
        network = network.cuda()
    network.load_checkpoint(checkpoint_name)

    agent_a = AlphaZeroAgent(train_spec.prediction_network(network), train_spec.game_engine(),
                             num_simulations=train_spec.num_simulations)
    print(f"Result: {Evaluation(train_spec.game_engine(), agent_a, UserAgent(), competitive=True).play()}")

    agent_a = AlphaZeroAgent(train_spec.prediction_network(network), train_spec.game_engine(),
                             num_simulations=train_spec.num_simulations)
    print(f"Result: {Evaluation(train_spec.game_engine(), UserAgent(), agent_a, competitive=True).play()}")

