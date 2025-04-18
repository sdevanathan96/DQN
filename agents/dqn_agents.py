import os

import torch
from agents.dqn_base_agent import DDQNBaseAgent
from models.model import CnnDQN, DuelDQN


class DDQNAgent(DDQNBaseAgent):
    def _create_model(self, state_shape, action_dim):
        """Create CNN-based DQN model"""
        return CnnDQN(state_shape, action_dim)

    def save_checkpoint(self, fr, output):
        """Overriden to use specific checkpoint name for DDQN"""
        checkpath = output + '/checkpoint_model'
        os.makedirs(checkpath, exist_ok=True)
        torch.save({
            'frames': fr,
            'model': self.model.state_dict()
        }, '%s/checkpoint_ddqn_fr_%d.tar'% (checkpath, fr))


class DuelDQNAgent(DDQNBaseAgent):
    def _create_model(self, state_shape, action_dim):
        """Create Dueling DQN model"""
        return DuelDQN(state_shape, action_dim)
    
    def save_checkpoint(self, fr, output):
        """Overriden to use specific checkpoint name for Dueling DQN"""
        checkpath = output + '/checkpoint_model'
        os.makedirs(checkpath, exist_ok=True)
        torch.save({
            'frames': fr,
            'model': self.model.state_dict()
        }, '%s/checkpoint_duel_fr_%d.tar'% (checkpath, fr))


class DQNAgent(DDQNAgent):
    def save_checkpoint(self, fr, output):
        """Overriden to use specific checkpoint name for standard DQN"""
        checkpath = output + '/checkpoint_model'
        os.makedirs(checkpath, exist_ok=True)
        torch.save({
            'frames': fr,
            'model': self.model.state_dict()
        }, '%s/checkpoint_dqn_fr_%d.tar'% (checkpath, fr))

    def learning(self, fr):
        """Perform standard DQN learning step"""
        s0, a, r, s1, done = self.buffer.sample(self.config.batch_size)
        
        s0 = torch.tensor(s0, dtype=torch.float)
        s1 = torch.tensor(s1, dtype=torch.float)
        a = torch.tensor(a, dtype=torch.long)
        r = torch.tensor(r, dtype=torch.float)
        done = torch.tensor(done, dtype=torch.float)
        
        if self.config.use_cuda:
            s0 = s0.cuda()
            s1 = s1.cuda()
            a = a.cuda()
            r = r.cuda()
            done = done.cuda()
        
        q_values = self.model(s0)
        
        with torch.no_grad():
            next_q_values = self.target_model(s1)
            next_q_value = next_q_values.max(1)[0]
        
        q_value = q_values.gather(1, a.unsqueeze(1)).squeeze(1)
        expected_q_value = r + self.config.gamma * next_q_value * (1 - done)
        
        loss = (q_value - expected_q_value.detach()).pow(2).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if fr % self.config.update_tar_interval == 0:
            self.target_model.load_state_dict(self.model.state_dict())
        
        return loss.item()