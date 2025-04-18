import os
import random

import torch
from agents.base_agent import BaseAgent
from buffer.buffer import ReplayBuffer
from utils.config import Config


class DDQNBaseAgent(BaseAgent):
    def __init__(self, config: Config):
        self.buffer = self._create_buffer(config.max_buff)
        super().__init__(config)
        
        # Create target model
        self.target_model = self._create_model(self.config.state_shape, self.config.action_dim)
        self.target_model.load_state_dict(self.model.state_dict())
        
        if self.config.use_cuda:
            self.target_model.cuda()
    
    def _create_buffer(self, max_size):
        """Create experience replay buffer"""
        return ReplayBuffer(max_size)
    
    def act(self, state, epsilon=None):
        """Select action using epsilon-greedy"""
        if epsilon is None: 
            epsilon = self.config.epsilon_min
        
        if random.random() > epsilon or not self.is_training:
            state = torch.tensor(state, dtype=torch.float).unsqueeze(0)
            if self.config.use_cuda:
                state = state.cuda()
            q_value = self.model.forward(state)
            action = q_value.max(1)[1].item()
        else:
            action = random.randrange(self.config.action_dim)
        return action
    
    def learning(self, fr):
        """Perform DQN learning step"""
        # Sample from buffer
        s0, a, r, s1, done = self.buffer.sample(self.config.batch_size)
        
        # Convert to tensors
        s0 = torch.tensor(s0, dtype=torch.float)
        s1 = torch.tensor(s1, dtype=torch.float)
        a = torch.tensor(a, dtype=torch.long)
        r = torch.tensor(r, dtype=torch.float)
        done = torch.tensor(done, dtype=torch.float)
        
        # Move to CUDA if needed
        if self.config.use_cuda:
            s0 = s0.cuda()
            s1 = s1.cuda()
            a = a.cuda()
            r = r.cuda()
            done = done.cuda()
        
        # Compute Q-values
        q_values = self.model(s0)
        next_q_values = self.model(s1)
        next_q_state_values = self.target_model(s1)
        
        # Make sure everything is on correct device
        if self.config.use_cuda:
            q_values = q_values.cuda()
            next_q_values = next_q_values.cuda()
            next_q_state_values = next_q_state_values.cuda()
        
        # Compute TD target (Double DQN)
        q_value = q_values.gather(1, a.unsqueeze(1)).squeeze(1)
        next_q_value = next_q_state_values.gather(1, next_q_values.max(1)[1].unsqueeze(1)).squeeze(1)
        expected_q_value = r + self.config.gamma * next_q_value * (1 - done)
        
        # Compute loss
        loss = (q_value - expected_q_value.detach()).pow(2).mean()
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network
        if fr % self.config.update_tar_interval == 0:
            self.target_model.load_state_dict(self.model.state_dict())
        
        return loss.item()
    
    def cuda(self):
        """Move models to CUDA"""
        super().cuda()
        self.target_model.cuda()
    
    def save_checkpoint(self, fr, output):
        """Save a checkpoint"""
        checkpath = output + '/checkpoint_model'
        os.makedirs(checkpath, exist_ok=True)
        torch.save({
            'frames': fr,
            'model': self.model.state_dict()
        }, '%s/checkpoint_fr_%d.tar'% (checkpath, fr))
    
    def load_checkpoint(self, model_path):
        """Load from a checkpoint"""
        checkpoint = torch.load(model_path)
        fr = checkpoint['frames']
        self.model.load_state_dict(checkpoint['model'])
        self.target_model.load_state_dict(checkpoint['model'])
        return fr
