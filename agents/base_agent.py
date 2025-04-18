from abc import ABC, abstractmethod

import torch
from torch.optim import Adam
from utils.config import Config
from core.util import get_class_attr_val


class BaseAgent(ABC):
    def __init__(self, config: Config):
        self.config = config
        self.is_training = True
        
        # Create model
        self.model = self._create_model(self.config.state_shape, self.config.action_dim)
        self.optimizer = self._create_optimizer()
        
        if self.config.use_cuda:
            self.cuda()
    
    @abstractmethod
    def _create_model(self, state_shape, action_dim):
        """Create the neural network model"""
        pass
    
    def _create_optimizer(self):
        """Create the optimizer"""
        return Adam(self.model.parameters(), lr=self.config.learning_rate)
    
    @abstractmethod
    def act(self, state, epsilon=None):
        """Select an action based on state"""
        pass
    
    @abstractmethod
    def learning(self, fr):
        """Perform a learning step"""
        pass
    
    def cuda(self):
        """Move models to CUDA"""
        self.model.cuda()
    
    def load_weights(self, model_path):
        """Load model weights from file"""
        model = torch.load(model_path)
        if 'model' in model:
            self.model.load_state_dict(model['model'])
        else:
            self.model.load_state_dict(model)
    
    def save_model(self, output, name=''):
        """Save model to file"""
        torch.save(self.model.state_dict(), '%s/model_%s.pkl' % (output, name))
    
    def save_config(self, output):
        """Save configuration to file"""
        with open(output + '/config.txt', 'w') as f:
            attr_val = get_class_attr_val(self.config)
            for k, v in attr_val.items():
                f.write(str(k) + " = " + str(v) + "\n")
    
    @abstractmethod
    def save_checkpoint(self, fr, output):
        """Save a checkpoint"""
        pass
    
    @abstractmethod
    def load_checkpoint(self, model_path):
        """Load from a checkpoint"""
        pass