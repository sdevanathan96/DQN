import ale_py
import gymnasium as gym
import argparse

import torch
from tester.tester import Tester
from common.wrappers import make_atari, wrap_deepmind, wrap_pytorch
from utils.config import Config

def create_parser():
    parser = argparse.ArgumentParser(description="Deep Reinforcement Learning for Atari Games")
    
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--train', action='store_true', 
                          help='Train a new model from scratch')
    mode_group.add_argument('--test', action='store_true', 
                          help='Test a trained model (requires --model-path)')
    mode_group.add_argument('--resume-training', dest='retrain', action='store_true', 
                          help='Resume training from a checkpoint (requires --model-path)')
    
    parser.add_argument('--environment', dest='env', default='PongNoFrameskip-v4', type=str, 
                      help='Atari environment name (default: PongNoFrameskip-v4)')
    
    parser.add_argument('--model-path', dest='model_path', type=str, 
                      help='Path to saved model for testing or resuming training')
    
    return parser

def run_algorithm(agent_class, trainer_class, config_modifier=None):
    parser = create_parser()
    args = parser.parse_args()
    
    gym.register_envs(ale_py)
    
    # Base configuration
    config = Config()
    config.env = args.env
    config.gamma = 0.99
    config.epsilon = 1
    config.epsilon_min = 0.01
    config.eps_decay = 30000
    config.frames = 2000000
    config.use_cuda = torch.cuda.is_available()
    config.learning_rate = 1e-4
    config.max_buff = 100000
    config.update_tar_interval = 1000
    config.batch_size = 32
    config.print_interval = 5000
    config.log_interval = 5000
    config.checkpoint = True
    config.checkpoint_interval = 500000
    config.win_reward = 18  # PongNoFrameskip-v4
    config.win_break = True
    
    if config_modifier:
        config = config_modifier(config)
    
    if args.test:
        env = make_atari(config.env, render='human')
    else:
        env = make_atari(config.env)
    env = wrap_deepmind(env)
    env = wrap_pytorch(env)
    
    config.action_dim = env.action_space.n
    config.state_shape = env.observation_space.shape
    
    agent = agent_class(config)
    
    if args.train:
        trainer = trainer_class(agent, env, config)
        trainer.train()
    elif args.test:
        if args.model_path is None:
            print('ERROR: Testing requires a model path. Use --model-path to specify.')
            exit(1)
        tester = Tester(agent, env, args.model_path)
        tester.test(visualize=True)
    elif args.retrain:
        if args.model_path is None:
            print('ERROR: Resuming training requires a checkpoint. Use --model-path to specify.')
            exit(1)
        checkpoint_frame = agent.load_checkpoint(args.model_path)
        trainer = trainer_class(agent, env, config)
        trainer.train(checkpoint_frame)