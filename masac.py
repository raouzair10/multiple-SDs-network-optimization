import torch as T
import torch.nn.functional as F
import numpy as np
from networks import ActorSAC, CriticSAC
from buffer import ReplayBuffer

device = T.device('cuda' if T.cuda.is_available() else 'cpu')

class MASACAgent():
    def __init__(self, alpha, beta, input_dims, tau, gamma=0.99, n_actions=2, max_size=1000000, batch_size=100, num_agents=1, reward_scale=2):
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.alpha = alpha
        self.input_dims = input_dims
        self.beta = beta
        self.num_agents = num_agents
        self.n_actions = n_actions
        self.scale = reward_scale

        self.memory = ReplayBuffer(max_size, input_dims, n_actions)

        self.critic_1 = CriticSAC(s_dim=input_dims, a_dim=n_actions).to(device)
        self.critic_2 = CriticSAC(s_dim=input_dims, a_dim=n_actions).to(device)
        self.target_critic_1 = CriticSAC(s_dim=input_dims, a_dim=n_actions).to(device)
        self.target_critic_2 = CriticSAC(s_dim=input_dims, a_dim=n_actions).to(device)

        self.critic_1_optimizer = T.optim.Adam(self.critic_1.parameters(), lr=beta)
        self.critic_2_optimizer = T.optim.Adam(self.critic_2.parameters(), lr=beta)

        self.actors = [ActorSAC(s_dim=input_dims, a_dim=1, alpha=alpha).to(device) for _ in range(num_agents)]
        self.actor_optimizers = [T.optim.Adam(actor.parameters(), lr=alpha) for actor in self.actors]

        self.update_network_parameters(tau=1)

    def choose_action(self, s, agent_index, evaluate=False):
        self.actors[agent_index].eval()
        s = T.FloatTensor(s.reshape(1, -1)).to(device)
        mean, log_std = self.actors[agent_index](s)
        std = log_std.exp()

        if evaluate:
            action = mean
        else:
            action = mean + std * T.randn_like(mean)

        action = T.tanh(action)  # Squash
        self.actors[agent_index].train()
        return agent_index, action.cpu().detach().numpy().flatten()

    def remember(self, state, action, reward, state_):
        self.memory.store_transition(state, action, reward, state_)

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        states, actions, rewards, next_states = self.memory.sample_buffer(self.batch_size)

        states = T.tensor(states, dtype=T.float).to(device)
        next_states = T.tensor(next_states, dtype=T.float).to(device)
        rewards = T.tensor(rewards, dtype=T.float).to(device)
        actions = T.tensor(actions, dtype=T.float).to(device)

        state_dim_per_agent = (states.shape[1] - 1) // self.num_agents
        shared_feature_idx = -1

        # Target actions
        next_actions = []
        next_log_probs = []

        for agent_idx in range(self.num_agents):
            start = agent_idx * state_dim_per_agent
            end = (agent_idx + 1) * state_dim_per_agent
            local_state = next_states[:, start:end]
            shared = next_states[:, shared_feature_idx].unsqueeze(1)
            actor_input = T.cat((local_state, shared), dim=1)

            mean, log_std = self.actors[agent_idx](actor_input)
            std = log_std.exp()
            noise = T.randn_like(std)
            action = T.tanh(mean + noise * std)

            log_prob = -0.5 * ((noise ** 2) + 2 * log_std + np.log(2 * np.pi))
            log_prob = log_prob.sum(dim=1, keepdim=True)

            next_actions.append(action)
            next_log_probs.append(log_prob)

        next_actions = T.cat(next_actions, dim=1)
        next_log_probs = T.cat(next_log_probs, dim=1).sum(dim=1, keepdim=True)

        q1_next = self.target_critic_1(next_states, next_actions)
        q2_next = self.target_critic_2(next_states, next_actions)
        q_next = T.min(q1_next, q2_next) - self.scale * next_log_probs

        q_target = rewards.unsqueeze(1) + self.gamma * q_next

        q1 = self.critic_1(states, actions)
        q2 = self.critic_2(states, actions)

        critic_1_loss = F.mse_loss(q1, q_target.detach())
        critic_2_loss = F.mse_loss(q2, q_target.detach())

        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()

        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        actor_losses = []
        for agent_idx in range(self.num_agents):
            start = agent_idx * state_dim_per_agent
            end = (agent_idx + 1) * state_dim_per_agent
            local_state = states[:, start:end]
            shared = states[:, shared_feature_idx].unsqueeze(1)
            actor_input = T.cat((local_state, shared), dim=1)

            mean, log_std = self.actors[agent_idx](actor_input)
            std = log_std.exp()
            noise = T.randn_like(std)
            action = T.tanh(mean + noise * std)

            log_prob = -0.5 * ((noise ** 2) + 2 * log_std + np.log(2 * np.pi))
            log_prob = log_prob.sum(dim=1, keepdim=True)

            joint_action = actions.clone()
            joint_action[:, agent_idx] = action.squeeze(1)

            q1_new = self.critic_1(states, joint_action)
            q2_new = self.critic_2(states, joint_action)
            q_new = T.min(q1_new, q2_new)

            actor_loss = (self.scale * log_prob - q_new).mean()
            actor_losses.append(actor_loss)

        total_actor_loss = T.stack(actor_losses).mean()

        for optimizer in self.actor_optimizers:
            optimizer.zero_grad()
        total_actor_loss.backward()
        for optimizer in self.actor_optimizers:
            optimizer.step()

        self.update_network_parameters()

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        for target_param, param in zip(self.target_critic_1.parameters(), self.critic_1.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        for target_param, param in zip(self.target_critic_2.parameters(), self.critic_2.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
