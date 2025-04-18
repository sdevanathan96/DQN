import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class CnnDQN(nn.Module):
    def __init__(self, inputs_shape, num_actions):
        super(CnnDQN, self).__init__()

        self.inut_shape = inputs_shape
        self.num_actions = num_actions

        self.features = nn.Sequential(
            nn.Conv2d(inputs_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(self.features_size(), 512),
            nn.ReLU(),
            nn.Linear(512, self.num_actions)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def features_size(self):
        return self.features(torch.zeros(1, *self.inut_shape)).view(1, -1).size(1)


class DuelDQN(nn.Module):
    def __init__(self, in_shape, n_actions):
        self.inut_shape = in_shape
        self.num_actions = n_actions
        super().__init__()
        self.conv1 = nn.Conv2d(self.inut_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.Adv1 = nn.Linear(64 * 7 * 7, 128)
        self.Adv2 = nn.Linear(128, self.num_actions)
        self.Val1 = nn.Linear(64 * 7 * 7, 128)
        self.Val2 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, 1)

        adv = F.leaky_relu(self.Adv1(x))
        adv = self.Adv2(adv)

        val = F.leaky_relu(self.Val1(x))
        val = self.Val2(val)

        return val + (adv - adv.mean())

class ActorCriticNetwork(nn.Module):
    def __init__(self, state_shape, n_actions):
        super(ActorCriticNetwork, self).__init__()
        self.state_shape = state_shape
        self.n_actions = n_actions
        
        # Shared CNN feature extractor
        self.conv = nn.Sequential(
            nn.Conv2d(state_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        conv_out_size = self._get_conv_out(state_shape)
        
        self.actor = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )
        
        self.critic = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
    
    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))
    
    def forward(self, state):
        features = self.conv(state)
        features = features.view(features.size(0), -1)
        
        actor_output = self.actor(features)
        action_probs = F.softmax(actor_output, dim=1)
        
        critic_output = self.critic(features)
        
        return action_probs, critic_output