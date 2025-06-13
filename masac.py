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
        self.bn1 = nn.BatchNorm1d(fc1_dim)
        self.fc2 = nn.Linear(fc1_dim, fc2_dim)
        self.mean = nn.Linear(fc2_dim, n_actions)
        self.log_std = nn.Linear(fc2_dim, n_actions)

    def forward(self, state):
        x = self.fc1(state)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = T.clamp(log_std, min=-20, max=2)
        return mean, log_std

class Critic(nn.Module):
    def __init__(self, input_dims, fc1_dim=64, fc2_dim=64):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(input_dims, fc1_dim)
        self.fc2 = nn.Linear(fc1_dim, fc2_dim)
        self.q = nn.Linear(fc2_dim, 1)

    def forward(self, state, action):
        x = T.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_value = self.q(x)
        return q_value

class MASAC:
    def __init__(self, lr_a, lr_c, global_input_dims, tau, gamma=0.99, n_actions=2, max_size=1000000, batch_size=100, num_agents=2):
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.lr_a = lr_a
        self.lr_c = lr_c
        self.num_agents = num_agents
        self.n_actions = n_actions

        self.memory = ReplayBuffer(max_size, global_input_dims, n_actions)

        critic_input_dims = global_input_dims + n_actions
        self.critic_1 = Critic(critic_input_dims).to(device)
        self.critic_2 = Critic(critic_input_dims).to(device)
        self.target_critic_1 = Critic(critic_input_dims).to(device)
        self.target_critic_2 = Critic(critic_input_dims).to(device)

        self.critic_1_optimizer = T.optim.Adam(self.critic_1.parameters(), lr=lr_c)
        self.critic_2_optimizer = T.optim.Adam(self.critic_2.parameters(), lr=lr_c)

        self.actors = [Actor(s_dim=4, n_actions=1).to(device) for _ in range(num_agents)]
        self.actor_optimizers = [T.optim.Adam(actor.parameters(), lr=lr_a) for actor in self.actors]

        self.log_alpha = T.zeros(1, requires_grad=True, device=device)
        self.alpha_optimizer = T.optim.Adam([self.log_alpha], lr=lr_a)
        self.target_entropy = -T.prod(T.tensor(1, dtype=T.float32, device=device)).item()

        self.update_network_parameters(tau=1)

    def choose_action(self, s, agent_index):
        self.actors[agent_index].eval()
        s = T.FloatTensor(s.reshape(1, -1)).to(device)
        mean, log_std = self.actors[agent_index](s)
        std = T.exp(log_std)
        normal = T.distributions.Normal(mean, std)
        x_t = normal.rsample()
        action = T.tanh(x_t)
        self.actors[agent_index].train()
        return action.cpu().data.numpy().flatten()

    def remember(self, state, action, reward, next_state):
        self.memory.store_transition(state, action, reward, next_state)

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        states, actions, rewards, next_states = self.memory.sample_buffer(self.batch_size)

        states = T.tensor(states, dtype=T.float32).to(device)
        next_states = T.tensor(next_states, dtype=T.float32).to(device)
        actions = T.tensor(actions, dtype=T.float32).to(device)
        rewards = T.tensor(rewards, dtype=T.float32).unsqueeze(1).to(device)

        # Critic update
        with T.no_grad():
            next_actions_joint, next_log_probs_joint = self._sample_joint_actions(next_states)
            q1_next = self.target_critic_1(next_states, next_actions_joint)
            q2_next = self.target_critic_2(next_states, next_actions_joint)
            min_q_next = T.min(q1_next, q2_next) - T.exp(self.log_alpha) * next_log_probs_joint

            q_target = rewards + self.gamma * min_q_next

        q1 = self.critic_1(states, actions)
        q2 = self.critic_2(states, actions)

        critic_1_loss = F.mse_loss(q1, q_target)
        critic_2_loss = F.mse_loss(q2, q_target)

        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()

        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        # Actor update
        actor_losses = []
        log_probs_total = []
        for agent_idx in range(self.num_agents):
            local_states = self._get_local_state_batch(states, agent_idx).to(device)

            mean, log_std = self.actors[agent_idx](local_states)
            std = T.exp(log_std)
            normal = T.distributions.Normal(mean, std)
            x_t = normal.rsample()
            current_agent_action = T.tanh(x_t)
            log_prob = self._calculate_log_prob(mean, log_std, current_agent_action)

            new_actions = actions.clone()
            batch_indices = T.arange(self.batch_size).to(device)
            new_actions[batch_indices, agent_idx] = current_agent_action.squeeze()

            q1_pi = self.critic_1(states, new_actions)
            q2_pi = self.critic_2(states, new_actions)
            min_q_pi = T.min(q1_pi, q2_pi)

            actor_loss = (T.exp(self.log_alpha) * log_prob - min_q_pi).mean()
            actor_losses.append(actor_loss)
            log_probs_total.append(log_prob.mean().detach())

            self.actor_optimizers[agent_idx].zero_grad()
            actor_loss.backward()
            T.nn.utils.clip_grad_norm_(self.actors[agent_idx].parameters(), max_norm=1.0) 
            self.actor_optimizers[agent_idx].step()

        if log_probs_total:
            log_probs_mean = T.stack(log_probs_total).mean()
            alpha_loss = -(self.log_alpha * (log_probs_mean + self.target_entropy)).mean()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        for target_param, param in zip(self.target_critic_1.parameters(), self.critic_1.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        for target_param, param in zip(self.target_critic_2.parameters(), self.critic_2.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def _get_local_state(self, global_state, agent_idx):
        start = agent_idx * 3
        local_state = np.concatenate((global_state[start:start+3], global_state[-1:]))
        return local_state

    def _get_local_state_batch(self, states, agent_idx):
        start = agent_idx * 3
        local_states = T.cat((states[:, start:start+3], states[:, -1:]), dim=1)
        return local_states

    def _sample_joint_actions(self, next_states):
        next_actions_joint = []
        log_probs_joint = []
        for agent_idx in range(self.num_agents):
            local_state = self._get_local_state_batch(next_states, agent_idx).to(device)
            mean, log_std = self.actors[agent_idx](local_state)
            std = T.exp(log_std)
            normal = T.distributions.Normal(mean, std)
            x_t = normal.rsample()
            next_action = T.tanh(x_t)
            log_prob = self._calculate_log_prob(mean, log_std, next_action)
            next_actions_joint.append(next_action)
            log_probs_joint.append(log_prob)

        next_actions_joint = T.cat(next_actions_joint, dim=1)
        log_probs_joint = T.cat(log_probs_joint, dim=1).sum(dim=1, keepdim=True)
        return next_actions_joint, log_probs_joint

    def _calculate_log_prob(self, mean, log_std, action):
        std = T.exp(log_std)
        normal = T.distributions.Normal(mean, std)
        log_prob = normal.log_prob(self._inverse_tanh(action))
        log_prob -= T.log(1 - action.pow(2) + 1e-6)
        return log_prob.sum(dim=-1, keepdim=True)

    def _inverse_tanh(self, y):
        return 0.5 * T.log((1 + y) / (1 - y) + 1e-6)