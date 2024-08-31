import numpy as np

class Buffer:
    def __init__(self, mem_size, input_dim, n_action,
                 augment_data=None, augment_rewards=None,
                 expert_data_ratio=0.1, augment_noise_ratio=0.1):
        self.mem_size = mem_size
        self.mem_cnt = 0
        self.augment_data = augment_data
        self.augment_rewards = augment_rewards
        self.expert_data_ratio = expert_data_ratio
        self.expert_data_cutoff = 0
        self.augment_noise_ratio = augment_noise_ratio

        self.state_mem = np.zeros((self.mem_size, input_dim))
        self.new_state_mem = np.zeros((self.mem_size, input_dim))
        self.action_mem = np.zeros((self.mem_size, n_action))
        self.reward_mem = np.zeros(self.mem_size)
        self.terminal_mem = np.zeros(self.mem_size, dtype=bool)

    def __len__(self):
        return self.mem_cnt
    
    def can_sample(self, batch_size):
        if self.mem_cnt > (batch_size * 500):
            return True
        else:
            return False

    def store_transitions(self, state, action, reward, state_, done):
        index = self.mem_cnt % self.mem_size

        self.state_mem[index] = state
        self.new_state_mem[index] = state_
        self.action_mem[index] = action
        self.reward_mem[index] = reward
        self.terminal_mem[index] = done
        self.mem_cnt += 1

    def sample_memory(self, batch_size):
        mem = min(self.mem_cnt, self.mem_size)
        if self.expert_data_ratio > 0:
            expert_data_quantity = int(batch_size * self.expert_data_ratio)
            random_batch = np.random.choice(mem, batch_size - expert_data_quantity)
            expert_batch = np.random.choice(self.expert_data_cutoff, expert_data_quantity)
            batch = np.concatenate((random_batch, expert_batch))
        else:
            batch = np.random.choice(mem, batch_size)
        
        states = self.state_mem[batch]
        states_ = self.new_state_mem[batch]
        actions = self.action_mem[batch]
        rewards = self.reward_mem[batch]
        dones = self.terminal_mem[batch]

        # data augmentation (add some noise to data to make training more stable) and 
        # reward augmentation (make rewards that gets numerically larger seem to make learning better)
        if self.augment_data:
            state_noise_std = self.augment_noise_ratio * np.mean(np.abs(states))
            action_noise_std = self.augment_noise_ratio * np.mean(np.abs(actions))
            # add dynamic noise
            states = states + np.random.normal(0, state_noise_std, states.shape)
            actions = actions + np.random.normal(0, action_noise_std, actions.shape)

        if self.augment_rewards:
            rewards = rewards * 100

        return states, actions, rewards, states_, dones
    
    def save_to_csv(self, filename):
        # capture and save data from human (save and load from csv)
        np.savez(filename,
                 state=self.state_mem[:self.mem_cnt],
                 action=self.action_mem[:self.mem_cnt],
                 reward=self.reward_mem[:self.mem_cnt],
                 state_=self.new_state_mem[:self.mem_cnt],
                 done=self.terminal_mem[:self.mem_cnt])
        print(f'saved {filename}')

    def load_from_csv(self, filename, expert_data=True):
        try:
            data = np.load(filename)
            self.mem_cnt = len(data['state'])
            self.state_mem[:self.mem_cnt] = data['state']
            self.action_mem[:self.mem_cnt] = data['action']
            self.reward_mem[:self.mem_cnt] = data['reward']
            self.new_state_mem[:self.mem_cnt] = data['state_']
            self.terminal_mem[:self.mem_cnt] = data['done']
            print(f'successfully loaded {filename} into memory')
            print(f'{self.mem_cnt} memories loaded')

            if expert_data:
                self.expert_data_cutoff = self.mem_cnt
    
        except:
            print(f'unable to load memory from {filename}')