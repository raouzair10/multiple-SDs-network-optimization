import torch as T
import torch.nn.functional as F
import numpy as np
from networks import Actor, Critic
from buffer import ReplayBuffer

device = T.device('cuda' if T.cuda.is_available() else 'cpu')

class MASAC:
    def __init__(self, alpha, beta, input_dims, tau, gamma=0.99, n_actions=2, max_size=1000000, batch_size=100, num_agents=2, reward_scale=2):
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.alpha = alpha
        self.beta = beta
        self.num_agents = num_agents
        self.n_actions = n_actions
        self.reward_scale = reward_scale

        self.memory = ReplayBuffer(max_size, input_dims, n_actions)

        # Centralized critics (2 Q-networks + targets)
        self.critic_1 = Critic(input_dims, n_actions).to(device)
        self.critic_2 = Critic(input_dims, n_actions).to(device)
        self.target_critic_1 = Critic(input_dims, n_actions).to(device)
        self.target_critic_2 = Critic(input_dims, n_actions).to(device)

        self.critic_1_optimizer = T.optim.Adam(self.critic_1.parameters(), lr=beta)
        self.critic_2_optimizer = T.optim.Adam(self.critic_2.parameters(), lr=beta)

        # Decentralized actors (one per agent)
        self.actors = [Actor(s_dim=4).to(device) for _ in range(num_agents)]
        self.actor_optimizers = [T.optim.Adam(actor.parameters(), lr=alpha) for actor in self.actors]

        # Entropy temperature (optional automatic tuning)
        self.log_alpha = T.zeros(1, requires_grad=True, device=device)
        self.alpha_optimizer = T.optim.Adam([self.log_alpha], lr=alpha)
        self.target_entropy = -1  # For continuous single-dimensional action

        self.update_network_parameters(tau=1)

    def choose_action(self, s, agent_index):
        self.actors[agent_index].eval()
        s = T.FloatTensor(s.reshape(1, -1)).to(device)
        a = self.actors[agent_index](s)
        self.actors[agent_index].train()
        return agent_index, a.cpu().data.numpy().flatten()

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
            next_actions, next_log_probs = self._sample_next_actions(next_states)
            q1_next = self.target_critic_1(next_states, next_actions)
            q2_next = self.target_critic_2(next_states, next_actions)
            min_q_next = T.min(q1_next, q2_next) - T.exp(self.log_alpha) * next_log_probs

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
            agent_mask = (actions[:, 0].int() == agent_idx)
            if agent_mask.sum() == 0:
                continue

            # Extract local states for this agent in batch
            local_states = self._get_local_state_batch(states, agent_idx)[agent_mask]
            local_states = local_states.to(device)

            # Compute actions & log probs
            curr_time_sharing = self.actors[agent_idx](local_states)
            log_prob = self._calculate_log_prob(curr_time_sharing)

            # Reconstruct joint actions
            joint_actions = actions.clone()
            joint_actions[agent_mask, 1] = curr_time_sharing.squeeze()

            # Evaluate Q-values for current actor policy
            q1_pi = self.critic_1(states[agent_mask], joint_actions[agent_mask])
            q2_pi = self.critic_2(states[agent_mask], joint_actions[agent_mask])
            min_q_pi = T.min(q1_pi, q2_pi)

            # Actor loss (SAC objective)
            actor_loss = (T.exp(self.log_alpha) * log_prob - min_q_pi).mean()
            actor_losses.append(actor_loss)
            log_probs_total.append(log_prob.mean().detach())

            self.actor_optimizers[agent_idx].zero_grad()
            actor_loss.backward()
            self.actor_optimizers[agent_idx].step()

        # Entropy temperature tuning (optional)
        if log_probs_total:
            log_probs_mean = T.stack(log_probs_total).mean()
            alpha_loss = -(self.log_alpha * (log_probs_mean + self.target_entropy)).mean()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

        self.update_network_parameters()

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        # Soft update for both target critics
        for target_param, param in zip(self.target_critic_1.parameters(), self.critic_1.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        for target_param, param in zip(self.target_critic_2.parameters(), self.critic_2.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def _get_local_state(self, global_state, agent_idx):
        # Extracts local state of active agent (4 features per agent)
        start = agent_idx * 3
        local_state = np.concatenate((global_state[start:start+3], global_state[-1:]))  # including hn-PD-BS
        return local_state

    def _get_local_state_batch(self, states, agent_idx):
        # Extracts local state for an entire batch for a given agent
        start = agent_idx * 3
        local_states = T.cat((states[:, start:start+3], states[:, -1:]), dim=1)
        return local_states

    def _sample_next_actions(self, next_states):
        next_actions = []
        log_probs = []
        for i in range(next_states.shape[0]):
            sd_idx = int(np.random.choice(self.num_agents))  # sample a random agent for next state
            local_state = self._get_local_state(next_states[i].cpu().numpy(), sd_idx)
            local_state_tensor = T.FloatTensor(local_state).unsqueeze(0).to(device)

            time_sharing = self.actors[sd_idx](local_state_tensor)
            log_prob = self._calculate_log_prob(time_sharing)

            # Construct action [sd_idx, time_sharing]
            next_action = T.tensor([sd_idx, time_sharing.item()], dtype=T.float32).to(device)

            next_actions.append(next_action)
            log_probs.append(log_prob)

        next_actions = T.stack(next_actions)
        log_probs = T.stack(log_probs).unsqueeze(1)
        return next_actions, log_probs

    def _calculate_log_prob(self, action):
        # Gaussian assumption with fixed variance (for simplicity)
        log_std = 0  # assuming unit variance for demonstration
        log_prob = -0.5 * (action.pow(2) + 2 * log_std + np.log(2 * np.pi))
        return log_prob.sum(dim=-1, keepdim=True)

    def save_models(self, prefix):
        for idx, actor in enumerate(self.actors):
            T.save(actor.state_dict(), f"{prefix}_actor_{idx}.pth")
        T.save(self.critic_1.state_dict(), f"{prefix}_critic1.pth")
        T.save(self.critic_2.state_dict(), f"{prefix}_critic2.pth")

    def load_models(self, prefix):
        for idx, actor in enumerate(self.actors):
            actor.load_state_dict(T.load(f"{prefix}_actor_{idx}.pth"))
        self.critic_1.load_state_dict(T.load(f"{prefix}_critic1.pth"))
        self.critic_2.load_state_dict(T.load(f"{prefix}_critic2.pth"))
