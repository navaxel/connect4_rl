# connect4_rl

Specific requirements: kaggle_environments, gym, torch

### General presentation

This project offers to train a reinforcement learning agent to play Connect Four. 

The method used for the training is Deep Q Learning.

The performances are assessed by simulating games against our agent and: \
-an agent playing randomly \
-an agent playing with the negamax strategy \
-an agent using a Monte Carlo Tree Search (MCTS) algorithm to choose an action 

### Scripts folder

Contains various python scripts used for the training of our agent and the MCTS agent.

### Notebooks for testing

The project also contains three notebooks: \
-one where the agent is only trained against a random player \
-one where the agent is only trained against a MCTS player \
-one where the agent firstly learns with a random player then against a copy of himself
