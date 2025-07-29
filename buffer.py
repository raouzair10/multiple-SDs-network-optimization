import numpy as np
import torch

################################## Replay Buffer ##################################
class ReplayBuffer():
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, input_shape))
        self.new_state_memory = np.zeros((self.mem_size, input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)

    def store_transition(self, state, action, reward, state_):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_state_memory[index] = state_

        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]

        return states, actions, rewards, states_
    
################################## Rollout Buffer ##################################
class RolloutBuffer:
    """
    Buffer to store trajectories during policy rollouts.
    Used to collect and batch experiences for learning.
    """
    def __init__(self, num_agents, buffer_size, global_state_dim, action_dim=2, device=None):
        self.num_agents = num_agents
        self.buffer_size = buffer_size
        self.device = device
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
            torch.tensor(np.array(self.global_states), dtype=torch.float32).to(self.device),
            torch.tensor(np.array(self.actions), dtype=torch.float32).to(self.device),
            torch.tensor(np.array(self.logprobs), dtype=torch.float32).to(self.device),
            torch.tensor(np.array(self.rewards), dtype=torch.float32).to(self.device),
            torch.tensor(np.array(self.is_terminals), dtype=torch.bool).to(self.device),
            torch.tensor(np.array(self.agent_indices), dtype=torch.long).to(self.device)
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

class MATD3ReplayBuffer:
    def __init__(self, max_size, state_dim, action_dim):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((max_size, state_dim))
        self.new_state_memory = np.zeros((max_size, state_dim))
        self.action_memory = np.zeros((max_size, action_dim))
        self.reward_memory = np.zeros(max_size)

    def store_transition(self, state, action, reward, state_):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]

        return states, actions, rewards, states_
