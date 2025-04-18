from common.common import run_algorithm
from agents.ppo_agent import PPOAgent
from trainer.ppo_trainer import PPOTrainer

def configure_ppo(config):
    """PPO-specific configuration"""
    config.gae_lambda = 0.95
    config.policy_clip = 0.1
    config.n_epochs = 4
    config.episodes_per_update = 4
    config.max_buff = 10000
    config.batch_size = 64
    config.print_interval = 10000
    config.learning_rate = 2.5e-4
    return config

if __name__ == '__main__':
    run_algorithm(PPOAgent, PPOTrainer, config_modifier=configure_ppo)