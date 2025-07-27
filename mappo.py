import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
import numpy as np

# Use GPU if available, otherwise fallback to CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Device set to: {device}")

################################## Rollout Buffer ##################################
class RolloutBuffer:
    """
    Buffer to store trajectories during policy rollouts.
    Used to collect and batch experiences for learning.
    """
    def __init__(self, num_agents, buffer_size, global_state_dim, action_dim=2):
        self.num_agents = num_agents
        self.buffer_size = buffer_size
        self.clear()

    def store(self, global_state, action, logprob, reward, is_terminal, agent_index):
        """
        Store one step of experience into the buffer.
        """
        self.global_states.append(global_state)
        self.actions.append(action)
        self.logprobs.append(logprob)
        self.rewards.append(reward)
        self.is_terminals.append(is_terminal)
        self.agent_indices.append(agent_index)

    def sample(self):
        """
        Convert lists to PyTorch tensors and move to device.
        Returns batched tensors for training.
        """
        return (
            torch.tensor(np.array(self.global_states), dtype=torch.float32).to(device),
            torch.tensor(np.array(self.actions), dtype=torch.float32).to(device),
            torch.tensor(np.array(self.logprobs), dtype=torch.float32).to(device),
            torch.tensor(np.array(self.rewards), dtype=torch.float32).to(device),
            torch.tensor(np.array(self.is_terminals), dtype=torch.bool).to(device),
            torch.tensor(np.array(self.agent_indices), dtype=torch.long).to(device)
        )

    def clear(self):
        """
        Reset the buffer for the next round of data collection.
        """
        self.global_states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.agent_indices = []

################################## Actor Network ##################################
class Actor(nn.Module):
    """
    Policy network for each agent — maps state to action.
    """
    def __init__(self, state_dim, action_dim, action_std_init=0.6):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Tanh()  # Ensures action output is bounded between [-1, 1]
        )
        # Initial variance for action sampling (used in exploration)
        self.action_var = torch.full((action_dim,), action_std_init ** 2, dtype=torch.float32).to(device)

    def forward(self, state):
        return self.actor(state)

################################## Centralized Critic Network ##################################
class Critic(nn.Module):
    """
    Centralized value function — takes joint state-action input for all agents.
    Used to compute the value of a given state-action pair.
    """
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.critic = nn.Sequential(
            nn.Linear(state_dim + action_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, state, actions):
        # Concatenate state and action inputs
        x = torch.cat([state, actions], dim=-1)
        return self.critic(x)

################################## MAPPO Agent ##################################
class MAPPO:
    """
    Multi-Agent Proximal Policy Optimization (MAPPO) implementation.
    Each agent has its own actor network, but all share a centralized critic.
    """
    def __init__(self, num_agents, local_state_dim, global_state_dim, action_dim, action_std_init=0.6):
        self.num_agents = num_agents
        self.action_dim = action_dim
        self.gamma = 0.99  # Discount factor
        self.eps_clip = 0.2  # PPO clipping parameter
        self.K_epochs = 5  # PPO optimization steps

        # Shared buffer for all agents
        self.buffer = RolloutBuffer(num_agents, buffer_size=10000, global_state_dim=global_state_dim, action_dim=action_dim * num_agents)

        # Initialize actor networks for each agent
        self.actors = nn.ModuleList([
            Actor(local_state_dim, action_dim, action_std_init).to(device)
            for _ in range(num_agents)
        ])
        # Old actors used for stable PPO ratio calculation
        self.actors_old = nn.ModuleList([
            Actor(local_state_dim, action_dim, action_std_init).to(device)
            for _ in range(num_agents)
        ])
        # Copy parameters from current to old actors
        for old, new in zip(self.actors_old, self.actors):
            old.load_state_dict(new.state_dict())

        # Centralized critic
        self.critic = Critic(global_state_dim, action_dim * num_agents).to(device)
        self.critic_old = Critic(global_state_dim, action_dim * num_agents).to(device)
        self.critic_old.load_state_dict(self.critic.state_dict())

        # Optimizers
        self.actor_optimizers = [torch.optim.Adam(actor.parameters(), lr=0.002) for actor in self.actors]
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=0.002)

        self.mse_loss = nn.MSELoss()
        self.action_std = action_std_init  # For exploration decay

    def select_action(self, local_state, agent_idx):
        """
        Select action using the old policy for the given agent.
        Samples from a Gaussian distribution.
        """
        state = torch.tensor(local_state, dtype=torch.float32).to(device)
        if not torch.isfinite(state).all():
            print(f"[ERROR] Non-finite values in state for agent {agent_idx}: {state}")
            return np.zeros(self.action_dim), 0.0

        mean = self.actors_old[agent_idx](state)
        if not torch.isfinite(mean).all():
            print(f"[ERROR] Actor mean NaN for agent {agent_idx}: {mean}, from state: {state}")
            return np.zeros(self.action_dim), 0.0

        # Construct Gaussian distribution
        cov_mat = torch.diag(self.actors_old[agent_idx].action_var).float().to(device)
        dist = MultivariateNormal(mean, covariance_matrix=cov_mat)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        return action.detach().cpu().numpy(), action_logprob.item()

    def store_transition(self, global_state, joint_action, joint_logprob, reward, done, agent_idx):
        """
        Store a transition into the buffer.
        """
        self.buffer.store(global_state, joint_action, joint_logprob, reward, done, agent_idx)

    def update(self):
        """
        Perform the MAPPO update using collected experiences.
        """
        # Sample from buffer
        states, actions, logprobs, rewards, dones, agent_indices = self.buffer.sample()

        # Compute discounted rewards (returns)
        returns = []
        discounted_sum = 0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done:
                discounted_sum = 0
            discounted_sum = reward + self.gamma * discounted_sum
            returns.insert(0, discounted_sum)
        returns = torch.tensor(returns, dtype=torch.float32).to(device)

        # Critic update
        values = self.critic(states, actions).view(-1)
        critic_loss = self.mse_loss(values, returns.detach())
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor update (each agent)
        for i in range(self.num_agents):
            # Extract data for agent i
            idx = (agent_indices == i).nonzero(as_tuple=True)[0]
            if len(idx) == 0:
                continue

            # Extract agent-specific state and actions
            local_states = torch.stack([
                torch.cat([states[b, i*3:i*3+3], states[b, -1:]]) for b in idx
            ])
            local_returns = returns[idx]
            local_actions = actions[idx][:, i].unsqueeze(1)
            old_logprobs = logprobs[idx]

            # New action distribution
            mean = self.actors[i](local_states)
            if not torch.isfinite(mean).all():
                print(f"[ERROR] Actor mean NaN during update for agent {i}: {mean}, from local_states: {local_states}")
                continue

            cov_mat = torch.diag(self.actors[i].action_var).float().to(device).expand(len(idx), -1, -1)
            dist = MultivariateNormal(mean, covariance_matrix=cov_mat)
            new_logprobs = dist.log_prob(local_actions)

            # Advantage estimation
            with torch.no_grad():
                baseline = self.critic_old(states[idx], actions[idx]).squeeze()
                advantages = local_returns - baseline
                if advantages.numel() > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # PPO loss
            ratio = torch.exp(new_logprobs - old_logprobs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            actor_loss = -torch.min(surr1, surr2).mean() - 0.01 * dist.entropy().mean()  # includes entropy bonus

            self.actor_optimizers[i].zero_grad()
            actor_loss.backward()
            self.actor_optimizers[i].step()

        # Sync old networks with current
        for i in range(self.num_agents):
            self.actors_old[i].load_state_dict(self.actors[i].state_dict())
        self.critic_old.load_state_dict(self.critic.state_dict())

        # Clear buffer for next rollout
        self.buffer.clear()

    def decay_action_std(self, decay_rate=0.01, min_std=0.1):
        """
        Linearly decay the action standard deviation to reduce exploration over time.
        """
        self.action_std = max(self.action_std - decay_rate, min_std)
        for actor in self.actors:
            actor.action_var = torch.full((self.action_dim,), self.action_std ** 2, dtype=torch.float32).to(device)
        for actor_old in self.actors_old:
            actor_old.action_var = torch.full((self.action_dim,), self.action_std ** 2, dtype=torch.float32).to(device)
