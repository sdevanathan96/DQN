import numpy as np


class TrainingStats:
    def __init__(self):
        self.losses = []
        self.all_rewards = []
        self.episode_reward = 0
        self.ep_num = 0
        self.total_frames = 0
    
    def add_loss(self, loss):
        self.losses.append(loss)
    
    def add_episode_reward(self, reward):
        self.all_rewards.append(reward)
        self.ep_num += 1
    
    def increment_reward(self, reward):
        self.episode_reward += reward
    
    def reset_episode_reward(self):
        self.episode_reward = 0
    
    def get_avg_reward(self, window=100):
        return float(np.mean(self.all_rewards[-window:])) if self.all_rewards else 0
    
    def increment_frames(self, frames=1):
        self.total_frames += frames


class CheckpointManager:
    def __init__(self, agent, output_dir, checkpoint_interval):
        self.agent = agent
        self.output_dir = output_dir
        self.checkpoint_interval = checkpoint_interval
        self.frames_since_checkpoint = 0
    
    def check_and_save(self, frame_count):
        self.frames_since_checkpoint += 1
        if self.frames_since_checkpoint >= self.checkpoint_interval:
            self.agent.save_checkpoint(frame_count, self.output_dir)
            self.frames_since_checkpoint = 0
            return True
        return False