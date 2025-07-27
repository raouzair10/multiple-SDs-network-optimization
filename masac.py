import torch as T
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from buffer import ReplayBuffer  # Custom replay buffer for experience storage

# Set device for training (GPU if available)
device = T.device('cuda' if T.cuda.is_available() else 'cpu')

################################## Actor Network ##################################
class Actor(nn.Module):
    """
    Actor network: maps local state to a probability distribution over actions.
    Outputs both mean and log standard deviation for continuous action sampling.
    """
    def __init__(self, s_dim, n_actions, fc1_dim=64, fc2_dim=64):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(s_dim, fc1_dim)
        self.ln1 = nn.LayerNorm(fc1_dim)  # Layer normalization helps stabilize training
        self.fc2 = nn.Linear(fc1_dim, fc2_dim)
        self.mean = nn.Linear(fc2_dim, n_actions)      # Output: mean of Gaussian
        self.log_std = nn.Linear(fc2_dim, n_actions)   # Output: log(std) of Gaussian

    def forward(self, state):
        x = F.relu(self.ln1(self.fc1(state)))
        x = F.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = T.clamp(self.log_std(x), min=-20, max=2)  # Prevents extremely high/low std
        return mean, log_std

################################## Centralized Critic Network ##################################
class Critic(nn.Module):
    """
    Centralized Critic network: takes global state and joint actions of all agents,
    and outputs a single Q-value.
    """
    def __init__(self, input_dims, fc1_dim=64, fc2_dim=64):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(input_dims, fc1_dim)
        self.fc2 = nn.Linear(fc1_dim, fc2_dim)
        self.q = nn.Linear(fc2_dim, 1)

    def forward(self, state, action):
        x = T.cat([state, action], dim=1)  # Concatenate state and action vectors
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.q(x)

################################## MASAC Agent ##################################
class MASAC:
    """
    Multi-Agent Soft Actor-Critic (MASAC) implementation.
    Each agent has its own actor but shares two centralized critics.
    """
    def __init__(self, lr_a, lr_c, global_input_dims, tau, gamma=0.99,
                 n_actions=2, max_size=1000000, batch_size=100, num_agents=2,
                 action_scale=1.0):
        # Hyperparameters
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.num_agents = num_agents
        self.n_actions = n_actions
        self.action_scale = action_scale

        # Experience replay buffer
        self.memory = ReplayBuffer(max_size, global_input_dims, n_actions)

        # Centralized critics and their target networks
        critic_input_dims = global_input_dims + n_actions
        self.critic_1 = Critic(critic_input_dims).to(device)
        self.critic_2 = Critic(critic_input_dims).to(device)
        self.target_critic_1 = Critic(critic_input_dims).to(device)
        self.target_critic_2 = Critic(critic_input_dims).to(device)

        self.critic_1_optimizer = T.optim.Adam(self.critic_1.parameters(), lr=lr_c)
        self.critic_2_optimizer = T.optim.Adam(self.critic_2.parameters(), lr=lr_c)

        # Initialize actor networks for each agent and their targets
        self.actors = [Actor(s_dim=4, n_actions=1).to(device) for _ in range(num_agents)]
        self.actor_targets = [Actor(s_dim=4, n_actions=1).to(device) for _ in range(num_agents)]
        self.actor_optimizers = [T.optim.Adam(actor.parameters(), lr=lr_a) for actor in self.actors]

        # Entropy temperature (alpha) tuning for each agent
        self.log_alphas = [T.zeros(1, requires_grad=True, device=device) for _ in range(num_agents)]
        self.alpha_optimizers = [T.optim.Adam([log_alpha], lr=lr_a) for log_alpha in self.log_alphas]
        self.target_entropy = -1.0  # Encourages sufficient exploration

        # Initially copy parameters to target networks
        self.update_network_parameters(tau=1)

    def choose_action(self, state, agent_index):
        """
        Sample an action from the current policy of the selected agent.
        Uses reparameterization trick with tanh squashing.
        """
        self.actors[agent_index].eval()
        state = T.FloatTensor(state.reshape(1, -1)).to(device)
        mean, log_std = self.actors[agent_index](state)
        std = T.exp(log_std)
        normal = T.distributions.Normal(mean, std)
        x_t = normal.rsample()
        action = T.tanh(x_t)
        self.actors[agent_index].train()
        return (action * self.action_scale).cpu().data.numpy().flatten()

    def remember(self, state, action, reward, next_state):
        """
        Store a transition in the replay buffer.
        """
        self.memory.store_transition(state, action, reward, next_state)

    def learn(self):
        """
        Perform a single learning update for the critics, actors, and alpha.
        """
        if self.memory.mem_cntr < self.batch_size:
            return

        # Sample mini-batch
        states, actions, rewards, next_states = self.memory.sample_buffer(self.batch_size)

        states = T.tensor(states, dtype=T.float32).to(device)
        next_states = T.tensor(next_states, dtype=T.float32).to(device)
        actions = T.tensor(actions, dtype=T.float32).to(device)
        rewards = T.tensor(rewards, dtype=T.float32).unsqueeze(1).to(device)

        # ---------- Critic Update ----------
        with T.no_grad():
            # Sample next joint actions using target actors
            next_actions_joint, next_log_probs_joint = self._sample_joint_actions(next_states, use_target=True)
            # Get target Q-values from both critics
            q1_next = self.target_critic_1(next_states, next_actions_joint)
            q2_next = self.target_critic_2(next_states, next_actions_joint)
            # Use minimum Q-value for conservative estimate, subtract entropy
            min_q_next = T.min(q1_next, q2_next) - T.exp(T.stack(self.log_alphas)).mean() * next_log_probs_joint
            q_target = rewards + self.gamma * min_q_next

        # Current Q-values
        q1 = self.critic_1(states, actions)
        q2 = self.critic_2(states, actions)

        # Critic loss and update
        critic_1_loss = F.mse_loss(q1, q_target)
        critic_2_loss = F.mse_loss(q2, q_target)

        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()

        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        # ---------- Actor & Alpha Update ----------
        for agent_idx in range(self.num_agents):
            local_states = self._get_local_state_batch(states, agent_idx).to(device)
            mean, log_std = self.actors[agent_idx](local_states)
            std = T.exp(log_std)
            normal = T.distributions.Normal(mean, std)
            x_t = normal.rsample()
            action = T.tanh(x_t)

            # Compute log prob under squashed Gaussian policy
            log_prob = normal.log_prob(x_t) - T.log(1 - action.pow(2) + 1e-6)
            log_prob = log_prob.sum(dim=1, keepdim=True)

            # Replace only the selected agent's action in the joint action
            new_actions = actions.clone()
            batch_indices = T.arange(self.batch_size).to(device)
            new_actions[batch_indices, agent_idx] = action.squeeze()

            # Estimate Q-values for new action
            q1_pi = self.critic_1(states, new_actions)
            q2_pi = self.critic_2(states, new_actions)
            min_q_pi = T.min(q1_pi, q2_pi)

            alpha = T.exp(self.log_alphas[agent_idx])
            actor_loss = (alpha * log_prob - min_q_pi).mean()

            # Backprop actor loss
            self.actor_optimizers[agent_idx].zero_grad()
            actor_loss.backward()
            T.nn.utils.clip_grad_norm_(self.actors[agent_idx].parameters(), 1.0)
            self.actor_optimizers[agent_idx].step()

            # ---------- Entropy Temperature (Alpha) Update ----------
            alpha_loss = -(self.log_alphas[agent_idx] * (log_prob + self.target_entropy).detach()).mean()
            self.alpha_optimizers[agent_idx].zero_grad()
            alpha_loss.backward()
            self.alpha_optimizers[agent_idx].step()

        # ---------- Target Network Soft Update ----------
        self.update_network_parameters()

    def update_network_parameters(self, tau=None):
        """
        Soft update of target networks using Polyak averaging.
        """
        tau = self.tau if tau is None else tau
        for target_param, param in zip(self.target_critic_1.parameters(), self.critic_1.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        for target_param, param in zip(self.target_critic_2.parameters(), self.critic_2.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        for actor, target in zip(self.actors, self.actor_targets):
            for target_param, param in zip(target.parameters(), actor.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def _get_local_state_batch(self, states, agent_idx):
        """
        Extracts the agent's local view of the state from the global state.
        Here it assumes 3 features per agent + 1 global feature.
        """
        start = agent_idx * 3
        return T.cat((states[:, start:start+3], states[:, -1:]), dim=1)

    def _sample_joint_actions(self, next_states, use_target=False):
        """
        Samples joint actions for all agents (used during critic updates).
        Optionally uses target actor networks.
        """
        next_actions_joint = []
        log_probs_joint = []

        for agent_idx in range(self.num_agents):
            local_state = self._get_local_state_batch(next_states, agent_idx).to(device)
            actor = self.actor_targets[agent_idx] if use_target else self.actors[agent_idx]
            mean, log_std = actor(local_state)
            std = T.exp(log_std)
            normal = T.distributions.Normal(mean, std)
            x_t = normal.rsample()
            action = T.tanh(x_t)

            log_prob = normal.log_prob(x_t) - T.log(1 - action.pow(2) + 1e-6)
            log_prob = log_prob.sum(dim=1, keepdim=True)

            next_actions_joint.append(action)
            log_probs_joint.append(log_prob)

        # Concatenate all agents' actions and log probs
        next_actions_joint = T.cat(next_actions_joint, dim=1)
        log_probs_joint = T.cat(log_probs_joint, dim=1).sum(dim=1, keepdim=True)

        return next_actions_joint, log_probs_joint
