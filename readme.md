# Alpha Zero PyTorch

An implementation of the AlphaZero architecture with a simple interface for adding new games and support for training models, adjusting hyperparameters checkpoints, and play.

### Currently supported games
There are implementations and trained models for Tic Tac Toe, Connect 4, and a simplified version of Hey, That's My Fish!. The models have been trained for varying amounts of time and exhibit different levels of skill, feel free to play around with them or continue training them.

### Adding a new game
To add a new game, implement the interfaces provided in the "interfaces" directory. Once that is done, model training can begin.

### Training a model
Begin by choosing hyperparameters in the class that implements the TrainingSpecification interface, then train a model by calling the train.train function with a TrainingSpecification instance for the desired game. An optional checkpoint may be provided to continue training an existing model.

Hyperparameters include the number of residual layers in the DCNN. Note that existing checkpoints can only be loaded with a TrainingSpecification that uses the same number of residual layers.

To test a model against an agent that chooses a random move each time, the train.evaluate_random function can be used. To test a model against a human player, the train.compete_with_user function can be used.

### CUDA
If a PyTorch compatible GPU is present on the system, set CUDA to True in utils.py. Otherwise set it to False. Checkpoints created using CUDA can be loaded on non-CUDA setups and vice-versa.