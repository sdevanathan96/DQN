from utils.config import Config
from trainer.base_trainer import BaseTrainer


class PPOTrainer(BaseTrainer):
    def __init__(self, agent, env, config: Config):
        super().__init__(agent, env, config)
        self.episodes_since_update = 0
    
    def select_action(self, state, frame_count):
        """Select action using PPO policy"""
        return self.agent.act(state)
    
    def train_step(self, frame_count):
        """Perform PPO update"""
        return self.agent.learning(frame_count)
    
    def should_update(self):
        """PPO updates after collecting episodes_per_update episodes"""
        return False
    
    def store_experience(self, state, action_result, reward, next_state, done):
        """Store experience in PPO buffer"""
        action, action_prob, value = action_result
        self.agent.buffer.add(state, action, action_prob, value, reward, done)
    
    def train_episode(self, frame_count):
        """Override to handle episode-based updates"""
        frame_count, is_solved = super().train_episode(frame_count)
        
        self.episodes_since_update += 1
        
        if self.episodes_since_update >= self.config.episodes_per_update:
            loss = self.train_step(frame_count)
            if loss is not None:
                self.stats.add_loss(loss)
            self.episodes_since_update = 0
        
        return frame_count, is_solved