from abc import ABC, abstractmethod
from utils.config import Config
from core.util import get_output_folder
from trainer.trainer_utils import CheckpointManager, TrainingStats


class BaseTrainer(ABC):
    def __init__(self, agent, env, config: Config):
        self.agent = agent
        self.env = env
        self.config = config
        self.stats = TrainingStats()
        
        self.outputdir = get_output_folder(self.config.output, self.config.env)
        self.agent.save_config(self.outputdir)
        
        self.checkpoint_manager = CheckpointManager(
            agent, self.outputdir, config.checkpoint_interval
        )
    
    @abstractmethod
    def select_action(self, state, frame_count):
        """Strategy for selecting an action"""
        pass
    
    @abstractmethod
    def train_step(self, frame_count):
        """Perform a single training step"""
        pass
    
    @abstractmethod
    def should_update(self):
        """Determine if the agent should be updated"""
        pass
    
    def print_status(self, frame_count):
        """Print training status"""
        avg_reward = self.stats.get_avg_reward(10)
        latest_loss = self.stats.losses[-1] if self.stats.losses else 0
        print(f"frames: {frame_count:5d}, reward: {avg_reward:5f}, "
              f"loss: {latest_loss:4f}, episode: {self.stats.ep_num:4d}")
    
    def check_solved(self):
        """Check if environment is solved"""
        if len(self.stats.all_rewards) < 100:
            return False
        
        avg_reward = self.stats.get_avg_reward(100)
        latest_reward = self.stats.all_rewards[-1]
        
        if avg_reward >= self.config.win_reward and latest_reward > self.config.win_reward:
            self.agent.save_model(self.outputdir, 'best')
            print(f'Ran {self.stats.ep_num} episodes, best 100-episodes average '
                  f'reward is {avg_reward:3f}. Solved after {self.stats.ep_num - 100} trials ✓')
            return True
        return False
    
    def train_episode(self, frame_count):
        """Run a single training episode"""
        state, _ = self.env.reset()
        done = False
        
        while not done:
            action_result = self.select_action(state, frame_count)
            
            if isinstance(action_result, tuple):
                action = action_result[0]
            else:
                action = action_result
                
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            
            self.store_experience(state, action_result, reward, next_state, done)
            
            state = next_state
            self.stats.increment_reward(reward)
            self.stats.increment_frames()
            frame_count += 1
            
            if self.should_update():
                loss = self.train_step(frame_count)
                if loss is not None:
                    self.stats.add_loss(loss)
            
            if frame_count % self.config.print_interval == 0:
                self.print_status(frame_count)
            
            if self.config.checkpoint:
                self.checkpoint_manager.check_and_save(frame_count)
        
        self.stats.add_episode_reward(self.stats.episode_reward)
        self.stats.reset_episode_reward()
        
        return frame_count, self.check_solved()
    
    @abstractmethod
    def store_experience(self, state, action_result, reward, next_state, done):
        """Store experience in buffer"""
        pass
        
    def train(self, pre_fr=0):
        """Main training loop"""
        print("Starting Training....")
        frame_count = pre_fr
        is_solved = False
        
        while frame_count < self.config.frames and not is_solved:
            frame_count, is_solved = self.train_episode(frame_count)
            
            if is_solved and self.config.win_break:
                break
        
        if not is_solved:
            print(f'Did not solve after {self.stats.ep_num} episodes')
            self.agent.save_model(self.outputdir, 'last')
        
        return self.stats.all_rewards