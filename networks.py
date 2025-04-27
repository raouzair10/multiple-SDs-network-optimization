import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(0)

'''Actor and Critic Networks  for all the Algorithms'''

class Actor(nn.Module):
    def __init__(self, s_dim):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(s_dim, 64)
        self.l2 = nn.Linear(64, 64)
        self.a = nn.Linear(64, 1)

    def forward(self, s):
        x = F.relu(self.l1(s))
        x = F.relu(self.l2(x))
        a = torch.tanh(self.a(x))
        return a

class Critic(nn.Module):
    def __init__(self, s_dim, a_dim, trainable=True):
        super(Critic, self).__init__()
        self.W_s = nn.Parameter(torch.randn(s_dim, 64), requires_grad=trainable)
        self.W_a = nn.Parameter(torch.randn(a_dim, 64), requires_grad=trainable)
        self.b1 = nn.Parameter(torch.randn(1, 64), requires_grad=trainable)
        self.l1 = nn.Linear(64, 64)
        self.l2 = nn.Linear(64, 1)

    def forward(self, s, a):
        x = F.relu(torch.matmul(s, self.W_s) + torch.matmul(a, self.W_a) + self.b1)
        x = F.relu(self.l1(x))
        return self.l2(x)

import torch as T
import torch.nn as nn

class ActorSAC(nn.Module):
    def __init__(self, s_dim, a_dim, alpha=0.0003, fc1_dims=256, fc2_dims=256):
        super(ActorSAC, self).__init__()

        self.fc1 = nn.Linear(s_dim, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)

        self.mean = nn.Linear(fc2_dims, a_dim)
        self.log_std = nn.Linear(fc2_dims, a_dim)

        self.optimizer = T.optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = T.relu(self.fc1(state))
        x = T.relu(self.fc2(x))

        mean = self.mean(x)
        log_std = self.log_std(x)

        log_std = T.clamp(log_std, min=-20, max=2)  # Stability
        return mean, log_std


class CriticSAC(nn.Module):
    def __init__(self, s_dim, a_dim, beta=0.0003, fc1_dims=256, fc2_dims=256):
        super(CriticSAC, self).__init__()

        self.fc1 = nn.Linear(s_dim + a_dim, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.q = nn.Linear(fc2_dims, 1)

        self.optimizer = T.optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, action):
        x = T.cat([state, action], dim=1)
        x = T.relu(self.fc1(x))
        x = T.relu(self.fc2(x))
        q = self.q(x)
        return q
