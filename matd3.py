import torch as T
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from buffer import MATD3ReplayBuffer as ReplayBuffer
device = T.device('cuda' if T.cuda.is_available() else 'cpu')

################################## Actor Network ##################################

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
        return T.tanh(self.out(x))

################################## Centralized Critic Network ##################################

class Critic(nn.Module):
    def __init__(self, input_dims, fc1_dim=128, fc2_dim=128):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(input_dims, fc1_dim)
        self.fc2 = nn.Linear(fc1_dim, fc2_dim)
        self.q = nn.Linear(fc2_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.q(x)

################################## MATD3 Agent ##################################

class MATD3:
    def __init__(self, s_dim, a_dim, num_agents, lr_actor=1e-3, lr_critic=1e-3,
                 gamma=0.99, tau=0.01, max_size=100000, batch_size=64,
                 policy_noise=0.2, noise_clip=0.5, policy_delay=2):
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.num_agents = num_agents
        self.a_dim = a_dim
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.learn_step = 0

        self.memory = ReplayBuffer(max_size, s_dim, a_dim)

        self.actors = [Actor(s_dim=4, n_actions=1).to(device) for _ in range(num_agents)]
        self.actors_target = [Actor(s_dim=4, n_actions=1).to(device) for _ in range(num_agents)]
        self.actor_optimizers = [T.optim.Adam(actor.parameters(), lr=lr_actor) for actor in self.actors]

        critic_input_dim = s_dim + a_dim
        self.critic1 = Critic(critic_input_dim).to(device)
        self.critic2 = Critic(critic_input_dim).to(device)
        self.critic1_target = Critic(critic_input_dim).to(device)
        self.critic2_target = Critic(critic_input_dim).to(device)

        self.critic1_optimizer = T.optim.Adam(self.critic1.parameters(), lr=lr_critic)
        self.critic2_optimizer = T.optim.Adam(self.critic2.parameters(), lr=lr_critic)

        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        for i in range(num_agents):
            self.actors_target[i].load_state_dict(self.actors[i].state_dict())

    def choose_action(self, local_state, agent_idx, explore=False):
        self.actors[agent_idx].eval()
        state = T.FloatTensor(local_state).unsqueeze(0).to(device)
        action = self.actors[agent_idx](state)
        self.actors[agent_idx].train()

        action = ((action + 1) / 2).clamp(0, 1)
        action = action.cpu().data.numpy().flatten()

        if explore:
            action += np.random.normal(0, self.policy_noise, size=action.shape)
            action = np.clip(action, 0, 1)

        return action

    def remember(self, state, action, reward, next_state):
        self.memory.store_transition(state, action, reward, next_state)

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        self.learn_step += 1
        states, actions, rewards, next_states = self.memory.sample_buffer(self.batch_size)

        states = T.tensor(states, dtype=T.float32).to(device)
        next_states = T.tensor(next_states, dtype=T.float32).to(device)
        actions = T.tensor(actions, dtype=T.float32).to(device)
        rewards = T.tensor(rewards, dtype=T.float32).unsqueeze(1).to(device)

        with T.no_grad():
            next_actions = []
            for i in range(self.num_agents):
                local_next_state = self._get_local_state_batch(next_states, i)
                noise = (T.randn_like(local_next_state[:, :1]) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
                next_action = self.actors_target[i](local_next_state) + noise
                next_action = next_action.clamp(-1, 1)
                next_action = (next_action + 1) / 2
                next_actions.append(next_action.squeeze())
            next_actions = T.stack(next_actions, dim=1)
            next_input = T.cat([next_states, next_actions], dim=1)
            target_q1 = self.critic1_target(next_input)
            target_q2 = self.critic2_target(next_input)
            critic_target = rewards + self.gamma * T.min(target_q1, target_q2)

        current_q1 = self.critic1(T.cat([states, actions], dim=1))
        current_q2 = self.critic2(T.cat([states, actions], dim=1))

        critic1_loss = F.mse_loss(current_q1, critic_target)
        critic2_loss = F.mse_loss(current_q2, critic_target)

        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        if self.learn_step % self.policy_delay == 0:
            for i in range(self.num_agents):
                local_states = self._get_local_state_batch(states, i)
                pred_action = self.actors[i](local_states)
                pred_action = (pred_action + 1) / 2
                new_actions = actions.clone()
                new_actions[:, i] = pred_action.squeeze()
                actor_input = T.cat([states, new_actions], dim=1)
                actor_loss = -self.critic1(actor_input).mean()

                self.actor_optimizers[i].zero_grad()
                actor_loss.backward()
                self.actor_optimizers[i].step()

                self._soft_update(self.actors[i], self.actors_target[i])

            self._soft_update(self.critic1, self.critic1_target)
            self._soft_update(self.critic2, self.critic2_target)

    def _get_local_state_batch(self, states, agent_idx):
        start = agent_idx * 3
        return T.cat((states[:, start:start + 3], states[:, -1:]), dim=1)

    def _soft_update(self, net, target_net):
        for target_param, param in zip(target_net.parameters(), net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
