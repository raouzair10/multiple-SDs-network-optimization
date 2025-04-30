import torch as T
import torch.nn.functional as F
import numpy as np
from networks import Actor, Critic
from buffer import ReplayBuffer

device = T.device("cuda" if T.cuda.is_available() else "cpu")

class MADDPG():
    def __init__(self, alpha, beta, input_dims, tau, gamma=0.99, n_actions=2, max_size=1000000, batch_size=100, num_agents=1):
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.alpha = alpha
        self.beta = beta
        self.num_agents = num_agents
        self.n_actions = n_actions

        self.memory = ReplayBuffer(max_size, input_dims, n_actions)

        self.critic = Critic(s_dim=input_dims, a_dim=n_actions).to(device)
        self.target_critic = Critic(s_dim=input_dims, a_dim=n_actions).to(device)
        self.critic_optimizer = T.optim.Adam(self.critic.parameters(), lr=beta)

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

            ts_action = self.target_actors[agent_idx](actor_input)
            target_actions.append(ts_action)

        target_actions = T.cat(target_actions, dim=1)  

        # ------------ Critic Update ------------

        q_next = self.target_critic(next_states, target_actions).view(-1, 1)

        rewards = rewards.unsqueeze(1)

        # Total reward across agents (if needed)
        q_targets = rewards + self.gamma * q_next 

        q_eval = self.critic(states, actions).view(-1, 1)

        critic_loss = F.mse_loss(q_eval, q_targets.detach())
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ------------ Actor Updates ------------

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

            # Get the Q value from the critic for the joint action
            actor_q = self.critic(states, joint_action).view(-1, 1)

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

        # Update critic
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        # Update each actor
        for i in range(self.num_agents):
            for target_param, param in zip(self.target_actors[i].parameters(), self.actors[i].parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
