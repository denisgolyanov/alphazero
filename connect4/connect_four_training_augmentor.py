from connect4.connect_four_state import ConnectFourState

import torch

from self_play import TrainingExample


class ConnectFourTrainingAugmentor(object):
    def augment(self, training_example):
        assert isinstance(training_example, TrainingExample)

        result = []
        original_state = training_example._game_state
        num_cols = original_state._board.cols
        num_rows = original_state._board.rows

        flipped_board = original_state._board._clone()
        for col in range(num_cols):
            for row in range(num_rows):
                flipped_board[row, col] =  original_state._board[row, num_cols - col - 1]

        action_probability_tensor = torch.empty_like(training_example._action_probability_tesnor)
        assert action_probability_tensor.size()[0] == 1
        for col in range(num_cols):
            action_probability_tensor[0][col] = training_example._action_probability_tesnor[0][num_cols - col - 1]

        flipped_game_state = ConnectFourState(flipped_board, original_state.get_player())
        result.append(TrainingExample(flipped_game_state, action_probability_tensor))
        result[0]._value = training_example._value

        return result


