import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal, Categorical
import numpy as np

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Device set to: {device}")

################################## Rollout Buffer ##################################
class RolloutBuffer:
    def __init__(self, num_agents, buffer_size, global_state_dim, action_dim=2): # action_dim is 2
        self.num_agents = num_agents
        self.global_states = np.zeros([buffer_size, global_state_dim], dtype=np.float32)
        self.actions = np.zeros([buffer_size, action_dim], dtype=np.float32) # Store 2D actions
        self.logprobs = np.zeros([buffer_size], dtype=np.float32) # Store logprob of the active agent
        self.rewards = np.zeros([buffer_size], dtype=np.float32)
        self.is_terminals = np.zeros([buffer_size], dtype=np.bool_)
        self.agent_indices = np.zeros([buffer_size], dtype=np.int32) # Track active agent
        self.ptr = 0
        self.buffer_size = buffer_size

    def store(self, global_state, action, logprob, reward, is_terminal, agent_index):
        self.global_states[self.ptr] = global_state
        self.actions[self.ptr] = action
        self.logprobs[self.ptr] = logprob
        self.rewards[self.ptr] = reward
        self.is_terminals[self.ptr] = is_terminal
        self.agent_indices[self.ptr] = agent_index
        self.ptr = (self.ptr + 1) % self.buffer_size

    def sample(self, batch_size):
        indices = np.random.choice(self.buffer_size, size=batch_size, replace=False)
        return (
            self.global_states[indices],
            self.actions[indices],
            self.logprobs[indices],
            self.rewards[indices],
            self.is_terminals[indices],
            self.agent_indices[indices]
        )

    def clear(self):
        self.ptr = 0
        self.global_states[:] = 0
        self.actions[:] = 0
        self.logprobs[:] = 0
        self.rewards[:] = 0
        self.is_terminals[:] = False
        self.agent_indices[:] = 0

################################## Actor Network ##################################
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim=1, has_continuous_action_space=True, action_std_init=0.6): # action_dim = 1
        super(Actor, self).__init__()
        self.has_continuous_action_space = has_continuous_action_space
        self.action_dim = action_dim
        if has_continuous_action_space:
            self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)
            self.actor = nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, action_dim),
                nn.Tanh()
            )

    def forward(self, state):
        if self.has_continuous_action_space:
            return self.actor(state)
        else:
            return self.actor(state)
################################## Centralized Critic Network ##################################
class Critic(nn.Module):
    def __init__(self, state_dim_centralized, action_dim_total=2): # action_dim_total = 2
        super(Critic, self).__init__()
        self.critic = nn.Sequential(
            nn.Linear(state_dim_centralized + action_dim_total, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, state, all_actions):
        x = torch.cat([state, all_actions], dim=1)
        return self.critic(x)

################################## MAPPO Agent ##################################
class MAPPO:
    def __init__(self, num_agents, local_state_dim=4, global_state_dim=7, action_dim_per_agent=1, lr_actor=0.0002, lr_critic=0.0004, gamma=0.99, K_epochs=4, eps_clip=0.2, buffer_size=1000, has_continuous_action_space=True, action_std_init=0.6):
        self.num_agents = num_agents
        self.local_state_dim = local_state_dim
        self.global_state_dim = global_state_dim
        self.action_dim_per_agent = action_dim_per_agent # Output of each actor (scalar tsv)
        self.action_dim_buffer = 2 # Dimension of action stored in buffer
        self.has_continuous_action_space = has_continuous_action_space
        self.action_std_init = action_std_init
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.buffer_size = buffer_size
        self.buffer = RolloutBuffer(num_agents, buffer_size, global_state_dim, self.action_dim_buffer)

        self.actors = nn.ModuleList([Actor(local_state_dim, action_dim_per_agent, has_continuous_action_space, action_std_init).to(device) for _ in range(num_agents)])
        self.critics = Critic(global_state_dim, self.action_dim_buffer).to(device) # Critic takes global state and 2D action

        self.actor_optimizers = [torch.optim.Adam(actor.parameters(), lr=lr_actor) for actor in self.actors]
        self.critic_optimizer = torch.optim.Adam(self.critics.parameters(), lr=lr_critic)

        self.actors_old = nn.ModuleList([Actor(local_state_dim, action_dim_per_agent, has_continuous_action_space, action_std_init).to(device) for _ in range(num_agents)])
        for i in range(num_agents):
            self.actors_old[i].load_state_dict(self.actors[i].state_dict())
        self.critics_old = Critic(global_state_dim, self.action_dim_buffer).to(device)
        self.critics_old.load_state_dict(self.critics.state_dict())

        self.MseLoss = nn.MSELoss()

    def select_action(self, local_state, agent_index):
        actor = self.actors_old[agent_index]
        state = torch.FloatTensor(local_state).to(device)
        if self.has_continuous_action_space:
            action_mean = actor(state)
            action_var = torch.full((self.action_dim_per_agent,), self.action_std_init * self.action_std_init).to(device)
            cov_mat = torch.diag(action_var).unsqueeze(dim=0)
            dist = MultivariateNormal(action_mean, cov_mat)
            action = dist.sample()
            action_logprob = dist.log_prob(action)
        else:
            action_probs = actor(state)
            dist = Categorical(action_probs)
            action = dist.sample()
            action_logprob = dist.log_prob(action)
        return action.detach().cpu().numpy().flatten(), action_logprob.detach().cpu().numpy().flatten()

    def store_transition(self, global_state, action, logprob, reward, is_terminal, agent_index):
        self.buffer.store(global_state, action, logprob, reward, is_terminal, agent_index)

    def update(self):
        if self.buffer.ptr < self.buffer_size:
            return

        batch = self.buffer.sample(self.buffer_size)
        global_states, actions, old_logprobs, rewards, is_terminals, agent_indices = batch

        global_states = torch.tensor(global_states, dtype=torch.float32).to(device)
        actions = torch.tensor(actions, dtype=torch.float32).to(device)
        old_logprobs = torch.tensor(old_logprobs, dtype=torch.float32).to(device).squeeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device).squeeze(1)
        is_terminals = torch.tensor(is_terminals, dtype=torch.bool).to(device)
        agent_indices = torch.tensor(agent_indices, dtype=torch.long).to(device)

        # Optimize critic
        self.critic_optimizer.zero_grad()
        critic_value = self.critics(global_states, actions).squeeze(1)
        returns = np.zeros_like(rewards.cpu().numpy())
        for t in reversed(range(self.buffer_size)):
            discounted_reward = 0
            if not is_terminals[t]:
                discounted_reward = rewards[t] + self.gamma * discounted_reward
            else:
                discounted_reward = rewards[t]
            returns[t] = discounted_reward
        returns = torch.tensor(returns, dtype=torch.float32).to(device)
        critic_loss = self.MseLoss(critic_value, returns)
        critic_loss.backward()
        self.critic_optimizer.step()

        # Optimize actors
        for agent_id in range(self.num_agents):
            actor_optimizer = self.actor_optimizers[agent_id]
            actor_optimizer.zero_grad()

            agent_mask = (agent_indices == agent_id).nonzero(as_tuple=True)[0]
            if len(agent_mask) > 0:
                batch_indices = agent_mask
                local_states_agent = self._get_local_states(global_states[batch_indices], agent_id)
                current_actions_agent = actions[batch_indices]
                old_logprobs_agent = old_logprobs[batch_indices]

                actor = self.actors[agent_id]
                if self.has_continuous_action_space:
                    action_mean = actor(local_states_agent)
                    action_var = torch.full((self.action_dim_per_agent,), self.action_std_init ** 2).to(device)
                    cov_mat = torch.diag(action_var).unsqueeze(dim=0).expand_as(action_mean)
                    dist = MultivariateNormal(action_mean, cov_mat)
                    new_logprobs_agent = dist.log_prob(current_actions_agent[:, agent_id].unsqueeze(1)).squeeze()
                    entropy = dist.entropy().mean()
                else:
                    action_probs = actor(local_states_agent)
                    dist = Categorical(action_probs)
                    new_logprobs_agent = dist.log_prob(current_actions_agent[:, agent_id]).squeeze()
                    entropy = dist.entropy().mean()

                ratio = torch.exp(new_logprobs_agent - old_logprobs_agent[batch_indices].detach())
                advantages = returns[batch_indices] - self.critics_old(global_states[batch_indices], actions[batch_indices]).squeeze(1).detach()
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
                actor_loss = (-torch.min(surr1, surr2).mean() - 0.01 * entropy)
                actor_loss.backward()
                actor_optimizer.step()

        # Copy new weights into old policies
        for i in range(self.num_agents):
            self.actors_old[i].load_state_dict(self.actors[i].state_dict())
        self.critics_old.load_state_dict(self.critics.state_dict())

        self.buffer.clear()

    def _get_local_states(self, global_state_batch, agent_index):
        local_states = []
        for gs in global_state_batch:
            start_index = agent_index * 3
            local_state_part = gs[start_index:start_index + 3]
            shared_feature = gs[-1]
            local_state = torch.cat([local_state_part, shared_feature.unsqueeze(0)], dim=-1)
            local_states.append(local_state)
        return torch.stack(local_states)