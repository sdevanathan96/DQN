import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from agents.agent_utils import ParameterNoise
from agents.base_agent import BaseAgent
from buffer.buffer import PPOMemory
from utils.config import Config
from models.model import ActorCriticNetwork


class PPOAgent(BaseAgent):
    def __init__(self, config: Config):
        self.buffer = PPOMemory(config.max_buff, config.state_shape)
        
        self.gamma = config.gamma
        self.gae_lambda = config.gae_lambda
        self.policy_clip = config.policy_clip
        self.n_epochs = config.n_epochs
        
        super().__init__(config)
        
        self.perturbed_model = self._create_model(self.config.state_shape, self.config.action_dim)
        self.perturbed_model.load_state_dict(self.model.state_dict())
        
        self.param_noise = ParameterNoise(
            initial_stddev=0.1,
            desired_action_stddev=0.1,
            adaptation_coefficient=1.01
        )
        
        self.action_noise_scale = 0.1
        self.action_noise_decay = 0.9999
        
        if self.config.use_cuda:
            self.perturbed_model.cuda()
    
    def _create_model(self, state_shape, action_dim):
        """Create actor-critic network for PPO"""
        return ActorCriticNetwork(state_shape, action_dim)
    
    def perturb_model(self):
        """Apply parameter noise to perturbed model"""
        params = self.model.state_dict()
        
        noisy_params = {}
        for name, param in params.items():
            if 'actor' in name:
                noise = torch.randn_like(param) * self.param_noise.current_stddev
                noisy_params[name] = param + noise
            else:
                noisy_params[name] = param.clone()
        
        self.perturbed_model.load_state_dict(noisy_params)
    
    def act(self, state, epsilon=None):
        """Select action using PPO policy (with noise during training)"""
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0)
        
        if self.config.use_cuda:
            state = state.cuda()
        
        if not self.is_training:
            with torch.no_grad():
                action_probs, value = self.model(state)
                action = torch.argmax(action_probs, dim=1).item()
            return action
        
        with torch.no_grad():
            action_probs, value = self.model(state)
            perturbed_probs, _ = self.perturbed_model(state)
            
            action_distance = torch.sqrt(torch.sum((action_probs - perturbed_probs) ** 2)).item()
            
            self.param_noise.adapt(action_distance)
            
            dist = Categorical(perturbed_probs)
            action = dist.sample().item()
            
            action_prob = action_probs[0, action].item()
            value = value.item()
        
        self.perturb_model()
        
        if np.random.random() < self.action_noise_scale:
            other_actions = [a for a in range(self.config.action_dim) if a != action]
            if other_actions:
                action = np.random.choice(other_actions)
        
        self.action_noise_scale *= self.action_noise_decay
        
        return action, action_prob, value
    
    def learning(self, fr):
        """Perform PPO learning step"""
        if self.buffer.size() < self.config.batch_size:
            return 0
        
        states, actions, old_probs, old_vals, rewards, dones, batch_indices = self.buffer.sample(self.config.batch_size)
        
        states = torch.tensor(states, dtype=torch.float)
        actions = torch.tensor(actions, dtype=torch.long)
        old_probs = torch.tensor(old_probs, dtype=torch.float)
        old_vals = torch.tensor(old_vals, dtype=torch.float).view(-1)
        
        if self.config.use_cuda:
            states = states.cuda()
            actions = actions.cuda()
            old_probs = old_probs.cuda()
            old_vals = old_vals.cuda()
        
        advantages, returns = self._compute_gae(rewards, old_vals.cpu().numpy(), dones)
        advantages = torch.tensor(advantages, dtype=torch.float)
        returns = torch.tensor(returns, dtype=torch.float)
        
        if self.config.use_cuda:
            advantages = advantages.cuda()
            returns = returns.cuda()
        
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        total_loss = 0
        for _ in range(self.n_epochs):
            for batch_idx in batch_indices:
                batch_states = states[batch_idx]
                batch_actions = actions[batch_idx]
                batch_old_probs = old_probs[batch_idx]
                batch_advantages = advantages[batch_idx]
                batch_returns = returns[batch_idx]
                
                action_probs, critic_value = self.model(batch_states)
                critic_value = critic_value.squeeze()
                
                dist = Categorical(action_probs)
                new_probs = dist.log_prob(batch_actions)
                
                entropy = dist.entropy().mean()
                
                ratio = torch.exp(new_probs - torch.log(batch_old_probs + 1e-10))
                
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.policy_clip, 1.0 + self.policy_clip) * batch_advantages
                
                actor_loss = -torch.min(surr1, surr2).mean()
                
                critic_loss = F.mse_loss(critic_value, batch_returns)
                
                loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy
                
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()
                
                total_loss += loss.item()
        
        self.perturb_model()
        
        self.buffer.clear()
        
        return total_loss / (self.n_epochs * len(batch_indices))
    
    def _compute_gae(self, rewards, values, dones):
        """
        Compute GAE (Generalized Advantage Estimation) advantage and returns,
        making sure to handle episode boundaries correctly.
        """
        advantages = np.zeros_like(rewards, dtype=np.float32)
        returns = np.zeros_like(rewards, dtype=np.float32)
        
        for i in range(len(self.buffer.traj_start_idx) - 1):
            start_idx = self.buffer.traj_start_idx[i]
            end_idx = self.buffer.traj_start_idx[i + 1]
            
            gae = 0
            
            for t in reversed(range(start_idx, end_idx)):
                if t == end_idx - 1:
                    next_value = 0
                    next_non_terminal = 0
                else:
                    next_value = values[t + 1]
                    next_non_terminal = 1.0 - dones[t]
                
                delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
                
                gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
                
                advantages[t] = gae
                returns[t] = gae + values[t]
        
        return advantages, returns
    
    def cuda(self):
        """Move models to CUDA"""
        super().cuda()
        self.perturbed_model.cuda()
    
    def load_weights(self, model_path):
        """Load model weights"""
        super().load_weights(model_path)
        
        self.perturbed_model.load_state_dict(self.model.state_dict())
        self.perturb_model()
    
    def save_checkpoint(self, fr, output):
        """Save checkpoint with noise parameters"""
        checkpath = output + '/checkpoint_model'
        os.makedirs(checkpath, exist_ok=True)
        torch.save({
            'frames': fr,
            'model': self.model.state_dict(),
            'param_noise_stddev': self.param_noise.current_stddev,
            'action_noise_scale': self.action_noise_scale
        }, '%s/checkpoint_ppo_fr_%d.tar'% (checkpath, fr))
    
    def load_checkpoint(self, model_path):
        """Load checkpoint with noise parameters"""
        checkpoint = torch.load(model_path)
        fr = checkpoint['frames']
        self.model.load_state_dict(checkpoint['model'])
        
        if 'param_noise_stddev' in checkpoint:
            self.param_noise.current_stddev = checkpoint['param_noise_stddev']
        if 'action_noise_scale' in checkpoint:
            self.action_noise_scale = checkpoint['action_noise_scale']
        
        self.perturbed_model.load_state_dict(self.model.state_dict())
        self.perturb_model()
        
        return fr