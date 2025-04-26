import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal, Categorical

'''Multi-Agent Proximal Policy Optimization Algorithm'''

# set device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Device set to: {device}")

################################## Rollout Buffer ##################################
class RolloutBuffer:
    def __init__(self, num_agents):
        self.num_agents = num_agents
        self.actions = [[] for _ in range(num_agents)]
        self.states = [[] for _ in range(num_agents)]
        self.logprobs = [[] for _ in range(num_agents)]
        self.rewards = [[] for _ in range(num_agents)]
        self.is_terminals = [[] for _ in range(num_agents)]

    def clear(self):
        for i in range(self.num_agents):
            self.actions[i].clear()
            self.states[i].clear()
            self.logprobs[i].clear()
            self.rewards[i].clear()
            self.is_terminals[i].clear()

################################## Actor Network ##################################
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, has_continuous_action_space, action_std_init):
        super(Actor, self).__init__()

        self.has_continuous_action_space = has_continuous_action_space

        if self.has_continuous_action_space:
            self.action_dim = action_dim
            self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)

            self.actor = nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, action_dim),
                nn.Tanh()
            )
        else:
            self.actor = nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, action_dim),
                nn.Softmax(dim=-1)
            )

    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)

    def act(self, state):
        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)

        return action.detach(), action_logprob.detach()

    def evaluate(self, state, action):
        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var)
            dist = MultivariateNormal(action_mean, cov_mat)

            action_logprobs = dist.log_prob(action)
            dist_entropy = dist.entropy()
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)

            action_logprobs = dist.log_prob(action)
            dist_entropy = dist.entropy()

        return action_logprobs, dist_entropy

################################## Centralized Critic Network ##################################
class CentralizedCritic(nn.Module):
    def __init__(self, state_dim, action_dim, num_agents):
        super(CentralizedCritic, self).__init__()

        self.critic = nn.Sequential(
            nn.Linear(num_agents * (state_dim + action_dim), 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, states, actions):
        x = torch.cat([states, actions], dim=-1)
        return self.critic(x)

################################## MAPPO ##################################
class MAPPO:
    def __init__(self, num_agents, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std_init=0.6):
        self.num_agents = num_agents
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.update_timestep = 1200
        self.has_continuous_action_space = has_continuous_action_space

        self.buffer = RolloutBuffer(num_agents)

        self.actors = [Actor(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device) for _ in range(num_agents)]
        self.optimizers = [torch.optim.Adam(actor.parameters(), lr=lr_actor) for actor in self.actors]

        self.critic = CentralizedCritic(state_dim, action_dim, num_agents).to(device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.MseLoss = nn.MSELoss()

    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            for actor in self.actors:
                actor.set_action_std(new_action_std)

    def select_action(self, state, agent_id):
        state = torch.FloatTensor(state).to(device)
        action, action_logprob = self.actors[agent_id].act(state)

        self.buffer.states[agent_id].append(state)
        self.buffer.actions[agent_id].append(action)
        self.buffer.logprobs[agent_id].append(action_logprob)

        return action.cpu().numpy(), action_logprob if self.has_continuous_action_space else action.item()

    def update(self):
        rewards = [[] for _ in range(self.num_agents)]
        discounted_reward = [0 for _ in range(self.num_agents)]

        for agent_id in range(self.num_agents):
            for reward, is_terminal in zip(reversed(self.buffer.rewards[agent_id]), reversed(self.buffer.is_terminals[agent_id])):
                if is_terminal:
                    discounted_reward[agent_id] = 0
                discounted_reward[agent_id] = reward + (self.gamma * discounted_reward[agent_id])
                rewards[agent_id].insert(0, discounted_reward[agent_id])

        # Normalize rewards
        normalized_rewards = []
        for rew in rewards:
            rew = torch.tensor(rew, dtype=torch.float32).to(device)
            normalized_rewards.append((rew - rew.mean()) / (rew.std() + 1e-7))

        # Convert buffer data to tensors
        old_states = [torch.stack(self.buffer.states[i]).detach() for i in range(self.num_agents)]
        old_actions = [torch.stack(self.buffer.actions[i]).detach() for i in range(self.num_agents)]
        old_logprobs = [torch.stack(self.buffer.logprobs[i]).detach() for i in range(self.num_agents)]

        # Joint State and Actions
        joint_states = torch.cat(old_states, dim=-1)
        joint_actions = torch.cat(old_actions, dim=-1)

        state_values = self.critic(joint_states, joint_actions).squeeze()

        advantages = []
        for i in range(self.num_agents):
            advantages.append(normalized_rewards[i] - state_values.detach())

        # Optimize policy and critic
        for agent_id in range(self.num_agents):
            for _ in range(self.K_epochs):
                logprobs, dist_entropy = self.actors[agent_id].evaluate(old_states[agent_id], old_actions[agent_id])

                ratios = torch.exp(logprobs - old_logprobs[agent_id])

                surr1 = ratios * advantages[agent_id]
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages[agent_id]

                loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, normalized_rewards[agent_id]) - 0.01 * dist_entropy

                self.optimizers[agent_id].zero_grad()
                loss.mean().backward()
                self.optimizers[agent_id].step()

        # Update centralized critic
        critic_loss = self.MseLoss(state_values, torch.mean(torch.stack(normalized_rewards), dim=0))
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # clear buffer
        self.buffer.clear()

    def save(self, checkpoint_path_prefix):
        for i, actor in enumerate(self.actors):
            torch.save(actor.state_dict(), f"{checkpoint_path_prefix}_actor_{i}.pth")
        torch.save(self.critic.state_dict(), f"{checkpoint_path_prefix}_critic.pth")

    def load(self, checkpoint_path_prefix):
        for i, actor in enumerate(self.actors):
            actor.load_state_dict(torch.load(f"{checkpoint_path_prefix}_actor_{i}.pth"))
        self.critic.load_state_dict(torch.load(f"{checkpoint_path_prefix}_critic.pth"))
