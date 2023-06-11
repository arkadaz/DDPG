import numpy as np

class ReplayBuffer():
    def __init__(self, max_size, input_shape, n_actions) -> None:
        self.memory_size = max_size
        self.memory_counter = 0
        self.state_memory = np.zeros((self.memory_size, *input_shape))
        self.next_state_memory = np.zeros((self.memory_size, *input_shape))
        self.action_memory = np.zeros((self.memory_size, n_actions))
        self.reward_memory = np.zeros(self.memory_size)
        self.terminal_memory = np.zeros(self.memory_size, dtype=np.bool_)

    def store_transition(self, state, action, reward, next_state, done):
        index = self.memory_counter % self.memory_size
        self.state_memory[index] = state
        self.next_state_memory[index] = next_state
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = done

        self.memory_counter += 1

    def sample_buffer(self, batch_size):
        max_memory = min(self.memory_counter, self.memory_size)

        batch = np.random.choice(max_memory, batch_size)
        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        next_states = self.next_state_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, next_states, dones
    