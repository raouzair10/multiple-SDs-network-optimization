import torch as T 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from buffer import ReplayBuffer  # Custom replay buffer for experience replay

# Set device to GPU if available
device = T.device('cuda' if T.cuda.is_available() else 'cpu')


################################## Actor Network ##################################

class Actor(nn.Module):
    """
    Actor network: maps local state to action.
    Used by each agent to decide what action to take.
    """
    def __init__(self, s_dim, n_actions, fc1_dim=64, fc2_dim=64):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(s_dim, fc1_dim)           # First hidden layer
        self.ln1 = nn.LayerNorm(fc1_dim)               # LayerNorm for stable learning
        self.fc2 = nn.Linear(fc1_dim, fc2_dim)         # Second hidden layer
        self.out = nn.Linear(fc2_dim, n_actions)       # Output layer (action)

    def forward(self, state):
        x = F.relu(self.ln1(self.fc1(state)))          # Apply LayerNorm + ReLU
        x = F.relu(self.fc2(x))
        return T.tanh(self.out(x))                     # Output in [-1, 1] range


################################## Centralized Critic Network ##################################

class Critic(nn.Module):
    """
    Centralized Critic: takes in all agent states and actions to estimate value.
    This is shared across agents (MADDPG setting).
    """
    def __init__(self, input_dims, fc1_dim=128, fc2_dim=128):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(input_dims, fc1_dim)
        self.fc2 = nn.Linear(fc1_dim, fc2_dim)
        self.v = nn.Linear(fc2_dim, 1)                 # Output scalar Q-value

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.v(x)                               # No activation (regression output)


################################## MADDPG Agent ##################################

class MADDPG:
    """
    Multi-Agent Deep Deterministic Policy Gradient (MADDPG) implementation.
    Supports multiple agents with independent actors and shared critic.
    """
    def __init__(self, s_dim, a_dim, num_agents, lr_actor=1e-3, lr_critic=1e-3,
                 gamma=0.99, tau=0.01, max_size=100000, batch_size=64):
        self.gamma = gamma                    # Discount factor
        self.tau = tau                        # Soft update rate
        self.batch_size = batch_size
        self.num_agents = num_agents
        self.a_dim = a_dim                    # Action dimension (usually 1)
        self.memory = ReplayBuffer(max_size, s_dim, a_dim)

        # Initialize actor and target actor networks per agent
        self.actors = [Actor(s_dim=4, n_actions=1).to(device) for _ in range(num_agents)]
        self.actors_target = [Actor(s_dim=4, n_actions=1).to(device) for _ in range(num_agents)]
        self.actor_optimizers = [T.optim.Adam(actor.parameters(), lr=lr_actor) for actor in self.actors]

        # Centralized critic shared by all agents
        self.critic = Critic(s_dim + a_dim).to(device)
        self.critic_target = Critic(s_dim + a_dim).to(device)
        self.critic_optimizer = T.optim.Adam(self.critic.parameters(), lr=lr_critic)

        # Copy initial weights from main to target networks
        self.critic_target.load_state_dict(self.critic.state_dict())
        for i in range(num_agents):
            self.actors_target[i].load_state_dict(self.actors[i].state_dict())

    def choose_action(self, local_state, agent_idx):
        """
        Chooses action for a specific agent based on its local state.
        Scales tanh output from [-1, 1] to [0, 1].
        """
        self.actors[agent_idx].eval()
        state = T.FloatTensor(local_state).unsqueeze(0).to(device)
        action = self.actors[agent_idx](state)
        self.actors[agent_idx].train()
        return ((action + 1) / 2).cpu().data.numpy().flatten()  # Convert to numpy, scaled to [0, 1]

    def remember(self, state, action, reward, next_state):
        """
        Store experience in replay buffer.
        """
        self.memory.store_transition(state, action, reward, next_state)

    def learn(self):
        """
        Perform learning update using sampled experience.
        Includes critic and actor updates and soft target updates.
        """
        if self.memory.mem_cntr < self.batch_size:
            return  # Not enough samples

        # Sample batch from buffer
        states, actions, rewards, next_states = self.memory.sample_buffer(self.batch_size)

        # Convert to tensors
        states = T.tensor(states, dtype=T.float32).to(device)
        next_states = T.tensor(next_states, dtype=T.float32).to(device)
        actions = T.tensor(actions, dtype=T.float32).to(device)
        rewards = T.tensor(rewards * 1000, dtype=T.float32).unsqueeze(1).to(device)  # Scaled rewards

        # Critic forward pass with current states and actions
        critic_input = T.cat([states, actions], dim=1)
        critic_value = self.critic(critic_input).squeeze()

        # Compute target Q-value using target networks
        with T.no_grad():
            next_actions = []
            for i in range(self.num_agents):
                local_next_state = self._get_local_state_batch(next_states, i)
                next_action = self.actors_target[i](local_next_state)
                next_action = (next_action + 1) / 2  # Rescale to [0, 1]
                next_actions.append(next_action.squeeze())

            next_actions = T.stack(next_actions, dim=1)
            next_input = T.cat([next_states, next_actions], dim=1)
            critic_target = rewards.squeeze() + self.gamma * self.critic_target(next_input).squeeze()

        # Critic loss and update
        critic_loss = F.mse_loss(critic_value, critic_target.detach())
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor update for each agent
        for i in range(self.num_agents):
            local_states = self._get_local_state_batch(states, i)
            pred_action = self.actors[i](local_states)
            pred_action = (pred_action + 1) / 2  # Rescale

            # Replace the i-th agent's action with its predicted action
            new_actions = actions.clone()
            new_actions[:, i] = pred_action.squeeze()

            actor_input = T.cat([states, new_actions], dim=1)
            actor_loss = -self.critic(actor_input).mean()  # Maximize Q-value

            self.actor_optimizers[i].zero_grad()
            actor_loss.backward()
            self.actor_optimizers[i].step()

            # Soft update actor target
            self._soft_update(self.actors[i], self.actors_target[i])

        # Soft update critic target
        self._soft_update(self.critic, self.critic_target)

    def _get_local_state_batch(self, states, agent_idx):
        """
        Extract local observation for a specific agent from the full state.
        Assumes local state = 3 features + 1 global feature = 4 total.
        """
        start = agent_idx * 3
        return T.cat((states[:, start:start+3], states[:, -1:]), dim=1)

    def _soft_update(self, net, target_net):
        """
        Soft update of target network parameters using Polyak averaging.
        """
        for target_param, param in zip(target_net.parameters(), net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
