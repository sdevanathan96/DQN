from common.common import run_algorithm
from agents.dqn_agents import DQNAgent
from trainer.dqn_trainer import DQNTrainer

def configure_ppo(config):
    return config

if __name__ == '__main__':
    run_algorithm(DQNAgent, DQNTrainer, config_modifier=configure_ppo)