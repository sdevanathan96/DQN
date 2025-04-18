from common.common import run_algorithm
from agents.dqn_agents import DDQNAgent
from trainer.dqn_trainer import DQNTrainer

def configure_ppo(config):
    return config

if __name__ == '__main__':
    run_algorithm(DDQNAgent, DQNTrainer, config_modifier=configure_ppo)