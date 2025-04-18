import math
from utils.config import Config
from trainer.base_trainer import BaseTrainer


class DQNTrainer(BaseTrainer):
    def __init__(self, agent, env, config: Config):
        super().__init__(agent, env, config)
        
        epsilon_final = self.config.epsilon_min
        epsilon_start = self.config.epsilon
        epsilon_decay = self.config.eps_decay
        self.epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(
            -1. * frame_idx / epsilon_decay)
    
    def select_action(self, state, frame_count):
        """Select action using epsilon-greedy"""
        epsilon = self.epsilon_by_frame(frame_count)
        return self.agent.act(state, epsilon)
    
    def train_step(self, frame_count):
        """Perform DQN update"""
        if self.agent.buffer.size() > self.config.batch_size:
            return self.agent.learning(frame_count)
        return 0
    
    def should_update(self):
        """DQN updates on every step if buffer is large enough"""
        return self.agent.buffer.size() > self.config.batch_size
    
    def store_experience(self, state, action, reward, next_state, done):
        """Store experience in DQN buffer"""
        self.agent.buffer.add(state, action, reward, next_state, done)