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

        # Shared centralized critic
        self.critic = Critic(s_dim=input_dims, a_dim=n_actions).to(device)
        self.target_critic = Critic(s_dim=input_dims, a_dim=n_actions).to(device)
        self.critic_optimizer = T.optim.Adam(self.critic.parameters(), lr=beta)

        # Per-agent actors
        self.actors = [Actor(s_dim=(input_dims // num_agents) + 1).to(device) for _ in range(num_agents)]
        self.target_actors = [Actor(s_dim=(input_dims // num_agents) + 1).to(device) for _ in range(num_agents)]
        self.actor_optimizers = [T.optim.Adam(actor.parameters(), lr=alpha) for actor in self.actors]

        # Sync targets
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

        # Sample a batch
        states, actions, rewards, next_states = self.memory.sample_buffer(self.batch_size)

        # Convert to PyTorch tensors
        states = T.tensor(states, dtype=T.float).to(device)        # shape: (batch_size, state_dim)
        next_states = T.tensor(next_states, dtype=T.float).to(device)
        rewards = T.tensor(rewards, dtype=T.float).to(device)      # shape: (batch_size,)
        actions = T.tensor(actions, dtype=T.float).to(device)      # shape: (batch_size, num_agents, 2) — if stored that way

        state_dim_per_agent = (states.shape[1] - 1) // self.num_agents  # excluding shared state
        shared_feature_idx = -1  # assuming the last feature is hn-PDi-BS

        # ----------- Target Actions ------------

        # Vectorize the target actions computation
        target_actions = []
        for agent_idx in range(self.num_agents):
            start = agent_idx * state_dim_per_agent
            end = (agent_idx + 1) * state_dim_per_agent

            local_state = next_states[:, start:end]              # (batch_size, local_state_dim)
            shared = next_states[:, shared_feature_idx].unsqueeze(1)  # (batch_size, 1)
            actor_input = T.cat((local_state, shared), dim=1)  # shape: (batch_size, local_state_dim + 1)

            ts_action = self.target_actors[agent_idx](actor_input)  # (batch_size, 1)
            target_actions.append(ts_action)

        target_actions = T.cat(target_actions, dim=1)  # shape: (batch_size, num_agents)

        # ------------ Critic Update ------------

        q_next = self.target_critic(next_states, target_actions).view(-1, 1)

        # Ensure rewards have the correct shape for sum (although it's not really needed since they're scalar)
        rewards = rewards.unsqueeze(1)  # shape: (batch_size, 1)

        # Total reward across agents (if needed)
        q_targets = rewards + self.gamma * q_next  # scalar rewards for each batch entry

        q_eval = self.critic(states, actions).view(-1, 1)

        critic_loss = F.mse_loss(q_eval, q_targets.detach())
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ------------ Actor Updates ------------

        # Vectorize the actor loss computation
        actor_losses = []
        for agent_idx in range(self.num_agents):
            # Get the local state and shared state
            start = agent_idx * state_dim_per_agent
            end = (agent_idx + 1) * state_dim_per_agent

            local_state = states[:, start:end]  # (batch_size, local_state_dim)
            shared = states[:, shared_feature_idx].unsqueeze(1)  # (batch_size, 1)
            actor_input = T.cat((local_state, shared), dim=1)  # (batch_size, local_state_dim + 1)

            # Get the predicted action from the actor for the batch
            new_actions = self.actors[agent_idx](actor_input)  # (batch_size, 1)

            # Rebuild the joint action
            joint_action = actions.clone()  # shape: (batch_size, num_agents, 2)
            joint_action[:, agent_idx] = new_actions.squeeze(1)  # Replace this agent's action

            # Get the Q value from the critic for the joint action
            actor_q = self.critic(states, joint_action).view(-1, 1)

            # Append the negative Q-value to the losses list
            actor_losses.append(-actor_q)

        # Calculate the mean of the actor losses
        actor_loss = T.stack(actor_losses).mean()

        # Backpropagation and optimizer step
        for agent_idx in range(self.num_agents):
            self.actor_optimizers[agent_idx].zero_grad()

        actor_loss.backward()
        for agent_idx in range(self.num_agents):
            self.actor_optimizers[agent_idx].step()

        # ------------ Soft Update Target Networks ------------

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
