import torch as T
import torch.nn.functional as F
import numpy as np
from networks import Actor, Critic
from buffer import ReplayBuffer

device = T.device("cuda" if T.cuda.is_available() else "cpu")

class MATD3():
    def __init__(self, alpha, beta, input_dims, tau, gamma=0.99, n_actions=2,
                 max_size=1000000, batch_size=100, num_agents=1,
                 policy_noise=0.2, noise_clip=0.5, policy_delay=2):
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.alpha = alpha
        self.beta = beta
        self.num_agents = num_agents
        self.n_actions = n_actions
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.learn_step = 0

        self.memory = ReplayBuffer(max_size, input_dims, n_actions)

        # Twin critics
        self.critic_1 = Critic(s_dim=input_dims, a_dim=n_actions).to(device)
        self.critic_2 = Critic(s_dim=input_dims, a_dim=n_actions).to(device)
        self.target_critic_1 = Critic(s_dim=input_dims, a_dim=n_actions).to(device)
        self.target_critic_2 = Critic(s_dim=input_dims, a_dim=n_actions).to(device)
        
        self.critic_optimizer_1 = T.optim.Adam(self.critic_1.parameters(), lr=beta)
        self.critic_optimizer_2 = T.optim.Adam(self.critic_2.parameters(), lr=beta)

        # Per-agent actors
        self.actors = [Actor(s_dim=(input_dims // num_agents) + 1).to(device) for _ in range(num_agents)]
        self.target_actors = [Actor(s_dim=(input_dims // num_agents) + 1).to(device) for _ in range(num_agents)]
        self.actor_optimizers = [T.optim.Adam(actor.parameters(), lr=alpha) for actor in self.actors]

        self.update_network_parameters(tau=1)

    def choose_action(self, s, agent_index):
        self.actors[agent_index].eval()
        s = T.FloatTensor(s.reshape(1, -1)).to(device)
        a = self.actors[agent_index](s)
        self.actors[agent_index].train()
        return a.cpu().data.numpy().flatten()

    def remember(self, state, action, reward, state_):
        self.memory.store_transition(state, action, reward, state_)

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        self.learn_step += 1

        states, actions, rewards, next_states = self.memory.sample_buffer(self.batch_size)

        states = T.tensor(states, dtype=T.float).to(device)
        next_states = T.tensor(next_states, dtype=T.float).to(device)
        rewards = T.tensor(rewards, dtype=T.float).to(device)
        actions = T.tensor(actions, dtype=T.float).to(device)

        state_dim_per_agent = (states.shape[1] - 1) // self.num_agents
        shared_feature_idx = -1

        target_actions = []
        for agent_idx in range(self.num_agents):
            start = agent_idx * state_dim_per_agent
            end = (agent_idx + 1) * state_dim_per_agent

            local_state = next_states[:, start:end]
            shared = next_states[:, shared_feature_idx].unsqueeze(1)
            actor_input = T.cat((local_state, shared), dim=1)

            next_action = self.target_actors[agent_idx](actor_input)

            # Add clipped noise
            noise = (T.randn_like(next_action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (next_action + noise).clamp(-1, 1) 

            target_actions.append(next_action)

        target_actions = T.cat(target_actions, dim=1)

        q1_next = self.target_critic_1(next_states, target_actions).view(-1, 1)
        q2_next = self.target_critic_2(next_states, target_actions).view(-1, 1)

        q_next = T.min(q1_next, q2_next)

        rewards = rewards.unsqueeze(1)
        q_target = rewards + self.gamma * q_next

        q1 = self.critic_1(states, actions).view(-1, 1)
        q2 = self.critic_2(states, actions).view(-1, 1)

        critic_loss_1 = F.mse_loss(q1, q_target.detach())
        critic_loss_2 = F.mse_loss(q2, q_target.detach())

        self.critic_optimizer_1.zero_grad()
        self.critic_optimizer_2.zero_grad()

        critic_loss_1.backward()
        critic_loss_2.backward()

        self.critic_optimizer_1.step()
        self.critic_optimizer_2.step()

        if self.learn_step % self.policy_delay == 0:
            actor_losses = []
            for agent_idx in range(self.num_agents):
                start = agent_idx * state_dim_per_agent
                end = (agent_idx + 1) * state_dim_per_agent

                local_state = states[:, start:end]
                shared = states[:, shared_feature_idx].unsqueeze(1)
                actor_input = T.cat((local_state, shared), dim=1)

                new_actions = self.actors[agent_idx](actor_input)

                joint_action = actions.clone()
                joint_action[:, agent_idx] = new_actions.squeeze(1)

                actor_q = self.critic_1(states, joint_action).view(-1, 1)
                actor_losses.append(-actor_q)

            actor_loss = T.stack(actor_losses).mean()

            for agent_idx in range(self.num_agents):
                self.actor_optimizers[agent_idx].zero_grad()
            actor_loss.backward()
            for agent_idx in range(self.num_agents):
                self.actor_optimizers[agent_idx].step()

            self.update_network_parameters()

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        for target_param, param in zip(self.target_critic_1.parameters(), self.critic_1.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        for target_param, param in zip(self.target_critic_2.parameters(), self.critic_2.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        for i in range(self.num_agents):
            for target_param, param in zip(self.target_actors[i].parameters(), self.actors[i].parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
