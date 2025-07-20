import torch as T
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from buffer import ReplayBuffer

device = T.device('cuda' if T.cuda.is_available() else 'cpu')

class Actor(nn.Module):
    def __init__(self, s_dim, n_actions, fc1_dim=64, fc2_dim=64):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(s_dim, fc1_dim)
        self.ln1 = nn.LayerNorm(fc1_dim)
        self.fc2 = nn.Linear(fc1_dim, fc2_dim)
        self.out = nn.Linear(fc2_dim, n_actions)

    def forward(self, state):
        x = F.relu(self.ln1(self.fc1(state)))
        x = F.relu(self.fc2(x))
        return T.tanh(self.out(x))  # Rescale if needed outside

class Critic(nn.Module):
    def __init__(self, input_dims, fc1_dim=128, fc2_dim=128):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(input_dims, fc1_dim)
        self.fc2 = nn.Linear(fc1_dim, fc2_dim)
        self.v = nn.Linear(fc2_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.v(x)

class MADDPG:
    def __init__(self, s_dim, a_dim, num_agents, lr_actor=1e-3, lr_critic=1e-3,
                 gamma=0.99, tau=0.01, max_size=100000, batch_size=64):
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.num_agents = num_agents
        self.a_dim = a_dim
        self.memory = ReplayBuffer(max_size, s_dim, a_dim)

        self.actors = [Actor(s_dim=4, n_actions=1).to(device) for _ in range(num_agents)]
        self.actors_target = [Actor(s_dim=4, n_actions=1).to(device) for _ in range(num_agents)]
        self.actor_optimizers = [T.optim.Adam(actor.parameters(), lr=lr_actor) for actor in self.actors]

        self.critic = Critic(s_dim + a_dim).to(device)
        self.critic_target = Critic(s_dim + a_dim).to(device)
        self.critic_optimizer = T.optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.critic_target.load_state_dict(self.critic.state_dict())
        for i in range(num_agents):
            self.actors_target[i].load_state_dict(self.actors[i].state_dict())

    def choose_action(self, local_state, agent_idx):
        self.actors[agent_idx].eval()
        state = T.FloatTensor(local_state).unsqueeze(0).to(device)
        action = self.actors[agent_idx](state)
        self.actors[agent_idx].train()
        return ((action + 1) / 2).cpu().data.numpy().flatten()  # Scale to [0, 1]

    def remember(self, state, action, reward, next_state):
        self.memory.store_transition(state, action, reward, next_state)

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        states, actions, rewards, next_states = self.memory.sample_buffer(self.batch_size)

        states = T.tensor(states, dtype=T.float32).to(device)
        next_states = T.tensor(next_states, dtype=T.float32).to(device)
        actions = T.tensor(actions, dtype=T.float32).to(device)
        rewards = T.tensor(rewards * 1000, dtype=T.float32).unsqueeze(1).to(device)

        critic_input = T.cat([states, actions], dim=1)
        critic_value = self.critic(critic_input).squeeze()

        # Get critic target
        with T.no_grad():
            next_actions = []
            for i in range(self.num_agents):
                local_next_state = self._get_local_state_batch(next_states, i)
                next_action = self.actors_target[i](local_next_state)
                next_action = (next_action + 1) / 2  # Rescale
                next_actions.append(next_action.squeeze())
            next_actions = T.stack(next_actions, dim=1)
            next_input = T.cat([next_states, next_actions], dim=1)
            critic_target = rewards.squeeze() + self.gamma * self.critic_target(next_input).squeeze()

        critic_loss = F.mse_loss(critic_value, critic_target.detach())
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        for i in range(self.num_agents):
            local_states = self._get_local_state_batch(states, i)
            pred_action = self.actors[i](local_states)
            pred_action = (pred_action + 1) / 2
            new_actions = actions.clone()
            new_actions[:, i] = pred_action.squeeze()
            actor_input = T.cat([states, new_actions], dim=1)
            actor_loss = -self.critic(actor_input).mean()

            self.actor_optimizers[i].zero_grad()
            actor_loss.backward()
            self.actor_optimizers[i].step()

            self._soft_update(self.actors[i], self.actors_target[i])

        self._soft_update(self.critic, self.critic_target)

    def _get_local_state_batch(self, states, agent_idx):
        start = agent_idx * 3
        return T.cat((states[:, start:start+3], states[:, -1:]), dim=1)

    def _soft_update(self, net, target_net):
        for target_param, param in zip(target_net.parameters(), net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
