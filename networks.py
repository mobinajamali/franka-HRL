import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import os

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
EPSILON = 1e-6

def weights_init_(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=1)
        nn.init.constant_(m.bias, 0)

class Actor(nn.Module):
    def __init__(self, input_dim, n_action, lr, fc1_dim, fc2_dim,
                 action_space=None, ckp_dir='tmp/', name='_actor_'):
        super(Actor, self).__init__()

        self.ckp_dir = ckp_dir
        self.ckp_file = os.path.join(ckp_dir, name)

        self.fc1 = nn.Linear(input_dim, fc1_dim)
        self.fc2 = nn.Linear(fc1_dim, fc2_dim)
        self.mu = nn.Linear(fc2_dim, n_action)
        self.sigma = nn.Linear(fc2_dim, n_action)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

        self.apply(weights_init_)

        # action rescaling (based on max action)
        if action_space is None:
            self.action_scale = T.tensor(1.).to(self.device)
            self.action_bias = T.tensor(0.).to(self.device)
        else:
            self.action_scale = T.FloatTensor((action_space.high - action_space.low) / 2).to(self.device)
            self.action_bias = T.FloatTensor((action_space.high + action_space.low) / 2).to(self.device)

    def forward(self, state):
        prob = F.relu(self.fc1(state))
        prob = F.relu(self.fc2(prob))
        mu = self.mu(prob)
        sigma = self.sigma(prob)
        sigma = T.clamp(sigma, min=LOG_SIG_MIN, max=LOG_SIG_MAX)

        return mu, sigma
    
    def sample_normal(self, state, reparametrization=True):
        mu, sigma = self.forward(state)
        probabilities = Normal(mu, sigma.exp()) 
        if reparametrization:
            actions = probabilities.rsample()
        else:
            actions = probabilities.sample()
        action = T.tanh(actions)*self.action_scale + self.action_bias
        log_prob = probabilities.log_prob(actions)
        log_prob -= T.log(self.action_scale*(1-action.pow(2))+EPSILON)
        log_prob = log_prob.sum(1, keepdim=True)
        mu = T.tanh(mu)*self.action_scale + self.action_bias

        return action, log_prob, mu
    
    def save_checkpoints(self):
        T.save(self.state_dict(), self.ckp_file)

    def load_checkpoints(self):
        self.load_state_dict(T.load(self.ckp_file))

class Critic(nn.Module):
    def __init__(self, input_dim, n_action, fc1_dim, fc2_dim,
                 lr, ckp_dir='tmp/', name='_critic_'):
        super(Critic, self).__init__()
        self.ckp_dir = ckp_dir
        self.ckp_file = os.path.join(self.ckp_dir, name)

        self.fc1 = nn.Linear(input_dim+n_action, fc1_dim)
        self.fc2 = nn.Linear(fc1_dim, fc2_dim)
        self.q = nn.Linear(fc2_dim, 1)

        self.fc3 = nn.Linear(input_dim+n_action, fc1_dim)
        self.fc4 = nn.Linear(fc1_dim, fc2_dim)
        self.q2 = nn.Linear(fc2_dim, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

        self.apply(weights_init_)

    
    def forward(self, state, action):
        #state = state.view(state.size(0), -1)
        #action = action.squeeze(1)
        #print(f"State shape: {state.shape}, Action shape: {action.shape}")
        combo = T.cat([state, action], dim=1)
        action = F.relu(self.fc1(combo))
        action = F.relu(self.fc2(action))
        q = self.q(action)

        action2 = F.relu(self.fc3(combo))
        action2 = F.relu(self.fc4(action2))
        q2 = self.q2(action2)

        return q, q2
    
    def save_checkpoints(self):
        T.save(self.state_dict(), self.ckp_file)

    def load_checkpoints(self):
        self.load_state_dict(T.load(self.ckp_file))
  