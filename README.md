This repository contains implementations of various deep reinforcement learning algorithms for playing Atari games, with a focus on Pong (can be extended to other games).

## Algorithms Implemented

- **DQN** (Deep Q-Network): The original DQN algorithm with experience replay and target networks
- **Double DQN**: Reduces overestimation bias by using separate networks for action selection and evaluation
- **Dueling DQN**: Uses a network architecture that separates state value and action advantage estimation
- **PPO** (Proximal Policy Optimization): On-policy algorithm with clipped surrogate objective and parameter noise for exploration

## Installation

### Prerequisites
```bash
pip install gymnasium[atari,accept-rom-license]
pip install ale-py
pip install torch numpy matplotlib
```

## Running the Algorithms

You can train, test, or resume training for any of the implemented algorithms.

### Training a New Model

```bash
# Train DQN on Pong
python atari_dqn.py --train --environment PongNoFrameskip-v4

# Train Double DQN on Pong
python atari_ddqn.py --train --environment PongNoFrameskip-v4

# Train Dueling DQN on Pong
python atari_dueldqn.py --train --environment PongNoFrameskip-v4

# Train PPO on Pong
python atari_ppo.py --train --environment PongNoFrameskip-v4
```
### Testing a Trained Model

```bash
# Test a trained DQN model
python atari_dqn.py --test --model-path sample_model/model_best_dqn.pkl --environment PongNoFrameskip-v4

# Test a trained PPO model
python atari_ppo.py --test --model-path sample_model/model_best_ppo.pkl --environment PongNoFrameskip-v4
```

### Resuming Training from a Checkpoint

```bash
# Resume DQN training from a checkpoint
python atari_dqn.py --resume-training --model-path path/to/checkpoint.tar
```

## Configuration Options
Each algorithm's behavior can be customized by modifying the config_modifier function in its respective entry script.
### Common Configuration Options
These options apply to all DQN algorithms:
```bash
config.gamma = 0.99                  # Discount factor
config.epsilon = 1                   # Initial exploration rate
config.epsilon_min = 0.01            # Minimum exploration rate
config.eps_decay = 30000             # Exploration decay rate
config.frames = 2000000              # Total frames to train
config.use_cuda = True               # Use GPU if available
config.learning_rate = 1e-4          # Learning rate
config.max_buff = 100000             # Replay buffer size
config.batch_size = 32               # Batch size
config.print_interval = 5000         # Steps between status prints
config.checkpoint_interval = 500000  # Steps between checkpoints
config.win_reward = 18               # Reward threshold for solving
```

### Algorithm-Specific Configuration
#### DQN, Double DQN, Dueling DQN
```bash
config.update_tar_interval = 1000    # Steps between target network updates
```
#### PPO Configuration
```bash
config.gae_lambda = 0.95             # GAE lambda parameter
config.policy_clip = 0.1             # PPO clipping parameter
config.n_epochs = 4                  # Number of optimization epochs
config.episodes_per_update = 4       # Episodes between policy updates
config.max_buff = 10000              # Buffer size
config.batch_size = 64               # Batch size
config.learning_rate = 2.5e-4        # Learning rate
```

### Customizing Configurations
You can modify an algorithm's behavior by editing the ``configure_ppo`` function in its entry script. For example, to change PPO's learning rate:
```bash
def configure_ppo(config):
    """PPO-specific configuration"""
    config.gae_lambda = 0.95
    config.policy_clip = 0.2         # Changed from 0.1 to 0.2
    config.n_epochs = 4
    config.episodes_per_update = 4
    config.max_buff = 10000
    config.batch_size = 64
    config.learning_rate = 3e-4      # Changed from 2.5e-4 to 3e-4
    return config
```
Similarly, for DQN Variants:
```bash
def configure_dqn(config):
    """DQN-specific configuration"""
    config.epsilon_min = 0.05        # Changed from 0.01 to 0.05
    config.eps_decay = 50000         # Changed from 30000 to 50000
    config.learning_rate = 5e-5      # Changed from 1e-4 to 5e-5
    return config

if __name__ == '__main__':
    run_algorithm(DQNAgent, DQNTrainer, config_modifier=configure_dqn)
```

## Acknowledgments

This implementation is based on the following research papers:

- Mnih et al., 2015 – *Human-level control through deep reinforcement learning*. [https://www.nature.com/articles/nature14236](https://www.nature.com/articles/nature14236)
- Schaul et al., 2016 – *Prioritized Experience Replay*. [https://arxiv.org/abs/1511.05952](https://arxiv.org/abs/1511.05952)
- Schulman et al., 2017 – *Proximal Policy Optimization Algorithms*. [https://arxiv.org/abs/1707.06347](https://arxiv.org/abs/1707.06347)
- Plappert et al., 2018 – *Parameter Space Noise for Exploration*. [https://arxiv.org/abs/1706.01905](https://arxiv.org/abs/1706.01905)
- Fortunato et al., 2018 – *Noisy Networks for Exploration*. [https://arxiv.org/abs/1706.10295](https://arxiv.org/abs/1706.10295)
- Fortunato et al., 2019 – *Effective Reinforcement Learning through Adaptive Noise Injection*. [https://arxiv.org/abs/1910.01417](https://arxiv.org/abs/1910.01417)
- Henderson et al., 2018 – *Deep Reinforcement Learning that Matters*. [https://arxiv.org/abs/1709.06560](https://arxiv.org/abs/1709.06560)
