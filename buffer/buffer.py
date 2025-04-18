import random
import numpy as np
from abc import ABC, abstractmethod


class BaseMemory(ABC):
    def __init__(self, capacity):
        self.capacity = capacity
    
    @abstractmethod
    def add(self, *args):
        """Add a transition to memory"""
        pass
    
    @abstractmethod
    def sample(self, batch_size):
        """Sample a batch of transitions"""
        pass
    
    @abstractmethod
    def size(self):
        """Get current size of memory"""
        pass
    
    @abstractmethod
    def clear(self):
        """Clear all memory"""
        pass


class ReplayBuffer(BaseMemory):
    def __init__(self, capacity):
        super().__init__(capacity)
        self.buffer = []
    
    def add(self, state, action, reward, next_state, done):
        """Add a transition to the buffer"""
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append((state[None, :], action, reward, next_state[None, :], done))
    
    def sample(self, batch_size):
        """Sample a random batch of transitions"""
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.concatenate(state), action, reward, np.concatenate(next_state), done
    
    def size(self):
        """Get current size of buffer"""
        return len(self.buffer)
    
    def clear(self):
        """Clear all transitions"""
        self.buffer = []


class PPOMemory(BaseMemory):
    def __init__(self, capacity, state_shape=None):
        super().__init__(capacity)
        self.states = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []
        self.counter = 0
        self.traj_start_idx = []
    
    def add(self, state, action, action_prob, value, reward, done):
        """Add a transition with policy information"""
        if len(self.states) == 0:
            self.traj_start_idx.append(0)
        
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(action_prob)
        self.vals.append(value)
        self.rewards.append(reward)
        self.dones.append(done)
        
        self.counter += 1
        
        if done:
            self.traj_start_idx.append(len(self.states))
    
    def sample(self, batch_size):
        """Return all data in batches (not random sampling)"""
        states = np.array(self.states)
        actions = np.array(self.actions)
        probs = np.array(self.probs)
        vals = np.array(self.vals)
        rewards = np.array(self.rewards)
        dones = np.array(self.dones)
        
        batch_indices = np.array_split(np.arange(len(states)), max(1, len(states) // batch_size))
        
        return states, actions, probs, vals, rewards, dones, batch_indices
    
    def size(self):
        """Get current size of memory"""
        return len(self.states)
    
    def clear(self):
        """Clear all memory"""
        self.states = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []
        self.traj_start_idx = []
        self.counter = 0


class MemoryFactory:
    @staticmethod
    def create_memory(memory_type, capacity, **kwargs):
        """Create memory of specified type"""
        if memory_type.lower() == 'replay':
            return ReplayBuffer(capacity)
        elif memory_type.lower() == 'ppo':
            state_shape = kwargs.get('state_shape', None)
            return PPOMemory(capacity, state_shape)
        else:
            raise ValueError(f"Unknown memory type: {memory_type}")


class OptimizedReplayBuffer(BaseMemory):
    def __init__(self, capacity, state_shape):
        super().__init__(capacity)
        self.state_shape = state_shape
        self.states = np.zeros((capacity, *state_shape), dtype=np.float32)
        self.next_states = np.zeros((capacity, *state_shape), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.bool_)
        self.position = 0
        self.size_count = 0
    
    def add(self, state, action, reward, next_state, done):
        """Add a transition to the buffer"""
        self.states[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_states[self.position] = next_state
        self.dones[self.position] = done
        
        self.position = (self.position + 1) % self.capacity
        self.size_count = min(self.size_count + 1, self.capacity)
    
    def sample(self, batch_size):
        """Sample a random batch of transitions"""
        batch_size = min(batch_size, self.size_count)
        indices = np.random.choice(self.size_count, batch_size, replace=False)
        
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices]
        )
    
    def size(self):
        """Get current size of buffer"""
        return self.size_count
    
    def clear(self):
        """Reset buffer"""
        self.position = 0
        self.size_count = 0