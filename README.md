# snake-RL
This project trains a Deep Q-Network (DQN) algorithm to play a snake game. The repository contains the game environment, DQN implementation, and utilities for training and visualizing results.

## Description

- **`env.py`**: Contains the snake game environment.
- **`agent.py`**: Contains the DQN training logic, reward calculation, and state representation.
- **`model.py`**: Contains the implementation of the DQN model.
- **`utils.py`**: Contains the plotting code for visualizing training progress.

## Dependencies

To run this project, you need the following Python packages:

- `torch`
- `pygame`
- `numpy`
- `matplotlib`
- `os`

You can install the required packages using pip:

```bash
pip install torch pygame numpy matplotlib