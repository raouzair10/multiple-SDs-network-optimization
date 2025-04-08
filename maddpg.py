import torch as T
import torch.nn.functional as F
from networks1 import Actor, Critic
from buffer1 import ReplayBuffer
import numpy as np

device = T.device("cuda" if T.cuda.is_available() else "cpu")

class Agent():
    def __init__(self, alpha, beta, input_dims, tau, gamma=0.99, n_actions=2, max_size=1000000, batch_size=100, num_agents=1):
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.alpha = alpha
        self.beta = beta
        self.num_agents = num_agents
        self.n_actions = n_actions

        self.memory = ReplayBuffer(max_size, input_dims, n_actions)

        self.actors = [Actor(s_dim=(input_dims//num_agents)+1) for _ in range(num_agents)] #list of actors
        self.critics = [Critic(s_dim=input_dims, a_dim=n_actions) for _ in range(num_agents)] #list of critics

        self.target_actors = [Actor(s_dim=(input_dims//num_agents)+1) for _ in range(num_agents)] #list of target actors
        self.target_critics = [Critic(s_dim=input_dims, a_dim=n_actions) for _ in range(num_agents)] #list of target critics

        self.actor_optimizers = [T.optim.Adam(actor.parameters(), lr=alpha) for actor in self.actors] #list of actor optimizers
        self.critic_optimizers = [T.optim.Adam(critic.parameters(), lr=beta) for critic in self.critics] #list of critic optimizers
        self.target_actor_optimizers = [T.optim.Adam(target_actor.parameters(), lr=alpha) for target_actor in self.target_actors] #list of target actor optimizers
        self.target_critic_optimizers = [T.optim.Adam(target_critic.parameters(), lr=beta) for target_critic in self.target_critics] #list of target critic optimizers

        for i in range(num_agents):
            self.update_network_parameters(tau=1, agent_index=i)

    def choose_action(self, s):
        sd_qualities = []
        for sd_index in range(self.num_agents):
            sd = []
            sd.append(s[0][sd_index * 3])  # hn-SDj-PDi
            sd.append(s[0][sd_index * 3 + 1])  # hn-SDj-BS
            sd.append(s[0][sd_index * 3 + 2]) # battery level

            sd = np.array(sd)
            min_val = np.min(sd)
            max_val = np.max(sd)

            sd = (sd - min_val) / (max_val - min_val)
            sd_quality = np.sum(sd)
            sd_qualities.append(sd_quality)

        # Choose SD with best quality
        if np.random.rand() < 0.1: # 10% random selection.
            best_sd_index = np.random.randint(self.num_agents - 1)
        else:
            best_sd_index = np.argmax(sd_qualities)

        self.actors[best_sd_index].eval()
        s_l = s[0][best_sd_index * (s.shape[1] // self.num_agents): (best_sd_index + 1) * (s.shape[1] // self.num_agents)]
        s_l = np.append(s_l, s[0][-1])  # [[hn-SDj-PDi hn-SDj-BS battery-SDj hn-PDi-BS]]
        s_local = T.FloatTensor(s_l.reshape(1, -1)).to(device)  # local state for each agent
        a = self.actors[best_sd_index](s_local)
        a = a.cpu().data.numpy().flatten()
        self.actors[best_sd_index].train()
        return best_sd_index, a

    def remember(self, state, action, reward, state_):
        self.memory.store_transition(state, action, reward, state_)

    def learn(self):
        if self.memory.mem_cntr < self.batch_size * 3:
            return

        states, actions, rewards, states_ = self.memory.sample_buffer(self.batch_size)

        states = T.tensor(states, dtype=T.float).to(device)
        states_ = T.tensor(states_, dtype=T.float).to(device)
        rewards = T.tensor(rewards, dtype=T.float).to(device)
        actions = T.tensor(actions, dtype=T.float).to(device)

        for agent_index in range(self.num_agents):  # learn each agent.
            # Target actions for the critic update
            target_actions = []
            for i in range(self.num_agents):
                local_next_state_part = states_.narrow(1, i * (states_.shape[1] // self.num_agents), (states_.shape[1] // self.num_agents))
                local_next_state = T.cat((local_next_state_part, states_[:, -1].unsqueeze(1)), dim=1)
                target_a = self.target_actors[i](local_next_state)
                target_actions.append(T.cat((T.ones(self.batch_size, 1, dtype=T.long).to(device) * i, target_a), dim=1))
            target_actions = T.cat(target_actions, dim=1)

            # Actions for the critic evaluation
            critic_value = self.critics[agent_index](states, actions)
            critic_value_ = self.target_critics[agent_index](states_, actions).view(-1)

            # Rewards are now accessed directly as it's a 1D tensor
            target = rewards + self.gamma * critic_value_
            target = target.view(self.batch_size, 1)

            self.critic_optimizers[agent_index].zero_grad()
            critic_loss = F.mse_loss(target, critic_value)
            critic_loss.backward()
            self.critic_optimizers[agent_index].step()

            # Actor loss
            local_state_part = states.narrow(1, agent_index * (states_.shape[1] // self.num_agents), (states_.shape[1] // self.num_agents))
            local_state = T.cat((local_state_part, states[:, -1].unsqueeze(1)), dim=1)
            current_actor_a = self.actors[agent_index](local_state)

            # Create the action input for the critic (SD index + time-sharing)
            actor_action_input = T.zeros(self.batch_size, self.n_actions).float().to(device)
            actor_action_input[:, 0] = agent_index  # SD index
            actor_action_input[:, 1] = current_actor_a.squeeze(1) # Time-sharing value

            actor_loss = -self.critics[agent_index](states, actor_action_input)
            actor_loss = T.mean(actor_loss)
            self.actor_optimizers[agent_index].zero_grad()
            actor_loss.backward()
            self.actor_optimizers[agent_index].step()

            self.update_network_parameters(agent_index=agent_index)

    def update_network_parameters(self, tau=None, agent_index = 0):
        if tau is None:
            tau = self.tau

        actor_params = self.actors[agent_index].named_parameters()
        critic_params = self.critics[agent_index].named_parameters()
        target_actor_params = self.target_actors[agent_index].named_parameters()
        target_critic_params = self.target_critics[agent_index].named_parameters()

        critic_state_dict = dict(critic_params)
        actor_state_dict = dict(actor_params)
        target_critic_state_dict = dict(target_critic_params)
        target_actor_state_dict = dict(target_actor_params)

        for name in critic_state_dict:
            critic_state_dict[name] = tau * critic_state_dict[name].clone() + (1 - tau) * target_critic_state_dict[name].clone()

        for name in actor_state_dict:
            actor_state_dict[name] = tau * actor_state_dict[name].clone() + (1 - tau) * target_actor_state_dict[name].clone()

        self.target_critics[agent_index].load_state_dict(critic_state_dict)
        self.target_actors[agent_index].load_state_dict(actor_state_dict)