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