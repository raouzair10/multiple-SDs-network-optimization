import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
import numpy as np

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Device set to: {device}")

################################## Rollout Buffer ##################################
class RolloutBuffer:
    def __init__(self, num_agents, local_state_dim, global_state_dim, action_dim, buffer_size):
        self.num_agents = num_agents
        self.local_state_dim = local_state_dim
        self.global_state_dim = global_state_dim
        self.action_dim = action_dim  # [sd_index, time_sharing_value]
        self.buffer_size = buffer_size
        self.ptr = 0
        self.path_slice = slice(0, buffer_size)

        self.global_states = np.zeros([buffer_size, global_state_dim], dtype=np.float32)
        self.actions = np.zeros([buffer_size, action_dim], dtype=np.float32)
        self.logprobs = np.zeros([buffer_size], dtype=np.float32)
        self.rewards = np.zeros([buffer_size], dtype=np.float32)
        self.is_terminals = np.zeros([buffer_size], dtype=np.bool_)
        self.agent_indices = np.zeros([buffer_size], dtype=np.int32) # To track which agent acted

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
class SimpleActor(nn.Module):
    def __init__(self, s_dim):
        super(SimpleActor, self).__init__()
        self.l1 = nn.Linear(s_dim, 64)
        self.l2 = nn.Linear(64, 64)
        self.a = nn.Linear(64, 1)

    def forward(self, s):
        x = F.relu(self.l1(s))
        x = F.relu(self.l2(x))
        a = torch.tanh(self.a(x))
        return a

################################## Centralized Critic Network ##################################
class SimpleCritic(nn.Module):
    def __init__(self, s_dim, a_dim):
        super(SimpleCritic, self).__init__()
        self.l1 = nn.Linear(s_dim + a_dim, 64)
        self.l2 = nn.Linear(64, 64)
        self.q = nn.Linear(64, 1)

    def forward(self, s, a):
        x = torch.cat([s, a], dim=-1)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        return self.q(x)

################################## MAPPO Agent ##################################
class MAPPO:
    def __init__(self, num_agents, local_state_dim, global_state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, buffer_size, action_std_init=0.6):
        self.num_agents = num_agents
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.buffer_size = buffer_size
        self.local_state_dim = local_state_dim
        self.global_state_dim = global_state_dim
        self.action_dim = action_dim # [sd_index, time_sharing_value]
        self.action_std_init = action_std_init

        self.buffer = RolloutBuffer(num_agents, local_state_dim, global_state_dim, action_dim, buffer_size)

        self.actors = [SimpleActor(local_state_dim).to(device) for _ in range(num_agents)]
        self.actor_optimizers = [torch.optim.Adam(actor.parameters(), lr=lr_actor) for actor in self.actors]

        self.critic = SimpleCritic(global_state_dim, action_dim).to(device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.MseLoss = nn.MSELoss()

    def get_local_state(self, global_state, agent_index):
        start_index = agent_index * 3  # Each SD has 3 local state features
        end_index = start_index + 3
        local_state_part = global_state[0, start_index:end_index]
        shared_feature = global_state[0, -1] # The last element is hn-PDi-BS
        local_state = torch.cat([local_state_part, shared_feature.unsqueeze(0)], dim=-1)
        return local_state

    def select_action(self, global_state, agent_index):
        local_state = self.get_local_state(global_state, agent_index)
        actor = self.actors[agent_index]
        action_ts_mean = actor(local_state.unsqueeze(0))
        action_var = torch.full_like(action_ts_mean, self.action_std_init ** 2).to(device)
        dist = MultivariateNormal(action_ts_mean, torch.diag_embed(action_var))
        action_ts = dist.sample()
        action_logprob = dist.log_prob(action_ts).squeeze()
        action = torch.zeros(self.action_dim).to(device)
        action[0] = agent_index # Store the active SD index
        action[1:] = action_ts.cpu() * 0.5 + 0.5 # Scale [-1, 1] to [0, 1]
        return action.cpu().numpy(), action_logprob.detach().numpy()

    def store_transition(self, global_state, action, logprob, reward, is_terminal, agent_index):
        self.buffer.store(global_state, action, logprob, reward, is_terminal, agent_index)

    def learn(self):
        if self.buffer.ptr < self.buffer_size: # Only learn when buffer is full
            return

        batch = self.buffer.sample(self.buffer_size)
        global_states, actions, logprobs, rewards, is_terminals, agent_indices = batch

        global_states = torch.tensor(global_states, dtype=torch.float32).to(device)
        actions = torch.tensor(actions, dtype=torch.float32).to(device)
        logprobs = torch.tensor(logprobs, dtype=torch.float32).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        is_terminals = torch.tensor(is_terminals, dtype=torch.bool).to(device)
        agent_indices = torch.tensor(agent_indices, dtype=torch.long).to(device)

        # Optimize critic
        self.critic_optimizer.zero_grad()
        critic_value = self.critic(global_states, actions).squeeze(1)
        returns = np.zeros_like(rewards)
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
                local_states_agent = torch.stack([self.get_local_state(global_states[i], agent_id) for i in batch_indices])
                old_logprobs_agent = logprobs[batch_indices]
                actions_agent = actions[batch_indices]

                actor = self.actors[agent_id]
                action_mean = actor(local_states_agent)
                action_var = torch.full_like(action_mean, self.action_std_init ** 2).to(device)
                dist = MultivariateNormal(action_mean, torch.diag_embed(action_var))
                new_logprobs_agent = dist.log_prob(actions_agent[:, 1:].unsqueeze(1)).squeeze()
                entropy = dist.entropy().mean()
                ratio = torch.exp(new_logprobs_agent - old_logprobs_agent.detach())

                values = self.critic(global_states[batch_indices], actions[batch_indices]).squeeze(1).detach()
                advantages = returns[batch_indices] - values

                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
                actor_loss = (-torch.min(surr1, surr2).mean() - 0.01 * entropy)
                actor_loss.backward()
                actor_optimizer.step()

        self.buffer.clear()

    def save(self, checkpoint_path_prefix):
        for i, actor in enumerate(self.actors):
            torch.save(actor.state_dict(), f"{checkpoint_path_prefix}_actor_{i}.pth")
        torch.save(self.critic.state_dict(), f"{checkpoint_path_prefix}_critic.pth")

    def load(self, checkpoint_path_prefix):
        for i, actor in enumerate(self.actors):
            actor.load_state_dict(torch.load(f"{checkpoint_path_prefix}_actor_{i}.pth"))
        self.critic.load_state_dict(torch.load(f"{checkpoint_path_prefix}_critic.pth"))