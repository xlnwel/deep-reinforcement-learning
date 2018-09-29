import numpy as np
import torch
import torch.nn.functional as F
from MyModel import *
import copy 
from collections import deque, namedtuple
import random
# import heapq

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class ReplayBuffer():
    def __init__(self, sample_size, max_len=int(1e5)):
        self.buffer = deque(maxlen=max_len)
        self.sample_size = sample_size
        self.experience = namedtuple('experience', ('state', 'action', 'reward', 'next_state', 'done'))

    def add(self, state, action, reward, next_state, done):
        exp = self.experience(state, action, reward, next_state, done)
        # here using exception handler to avoid unnecessary overhead in general
        # try:
        #     if len(self.buffer) <= self.max_len:
        #         heapq.heappush(self.buffer, [error, exp])
        #     else:
        #         heapq.heapreplace(self.buffer, [error, exp])
        # except(ValueError):
        #     errors = [item[0] for item in self.buffer]
        #     new_error_fn = lambda: error + random.uniform(1e-6, 1e-5)
        #     new_error = new_error_fn()
        #     while new_error in errors:
        #         new_error = new_error_fn()
        #     assert new_error not in errors, 'The error is already in the buffer.'
        #     if len(self.buffer) <= self.max_len:
        #         heapq.heappush(self.buffer, [new_error, exp])
        #     else:
        #         heapq.heapreplace(self.buffer, [new_error, exp])
        self.buffer.append(exp)

    def sample(self):
        # _, exps = zip(*random.sample(self.buffer, self.sample_size))
        exps = random.sample(self.buffer, self.sample_size)
        states = torch.from_numpy(np.vstack([e.state for e in exps if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in exps if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in exps if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in exps if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in exps if e is not None]).astype(np.uint8)).float().to(device)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


class Agent():
    def __init__(self, state_size, action_size, batch_size=64, 
                 actor_alpha=1e-4, critic_alpha=3e-4, gamma=0.9, tau=1e-3,
                 weight_decay=1e-4, actor_file=None, critic_file=None):
        # hyperparameters
        self.gamma = gamma
        self.tau = tau
        self.noise_decay = 1 + 5e-6
        # env info
        self.state_size = state_size
        self.action_size = action_size
        # replay buffer
        self.buffer = ReplayBuffer(sample_size=batch_size)
        # actor-critic
        self.actor_main = Actor(state_size, action_size, param_file=actor_file).to(device)
        self.critic_main = Critic(state_size, action_size, param_file=critic_file).to(device)
        # target actor-critic
        self.actor_target = copy.deepcopy(self.actor_main)
        self.critic_target = copy.deepcopy(self.critic_main)
        # optimizer
        self.actor_optimizer = torch.optim.Adam(self.actor_main.parameters(), lr=actor_alpha)
        self.critic_optimizer = torch.optim.Adam(self.critic_main.parameters(), lr=critic_alpha, weight_decay=weight_decay)

    def act(self, state):
        state = torch.from_numpy(state).float().to(device)
        # add noise to parameters
        saved_params = []
        for param in self.actor_main.parameters():
            saved_params.append(copy.deepcopy(param))
            param = param + torch.normal(mean=0.0, std=torch.ones_like(param) / (10 * self.noise_decay))

        self.noise_decay *= 1 + 5e-6

        self.actor_main.eval()
        with torch.no_grad():
           action = self.actor_main(state).cpu().numpy()
        self.actor_main.train()
        # restore parameters
        for param, saved_param in zip(self.actor_main.parameters(), saved_params):
            param = saved_param

        return np.clip(action, -1, 1)

    def step(self, state, action, reward, next_state, done):
        # compute error used as the priority number of priority queue
        # state_tensor = torch.from_numpy(np.reshape(state, (1, -1))).float().to(device)
        # action_tensor = torch.from_numpy(np.reshape(action, (1, -1))).float().to(device)
        # next_state_tensor = torch.from_numpy(np.reshape(next_state, (1, -1))).float().to(device)
        #
        # self.actor_main.eval()
        # self.critic_main.eval()
        # with torch.no_grad():
        #     value = self.critic_main(state_tensor, action_tensor).cpu().numpy()
        #     next_action_tensor = self.actor_main(next_state_tensor)
        #     next_value = self.critic_main(next_state_tensor, next_action_tensor).cpu().numpy()
        # self.actor_main.train()
        # self.critic_main.train()
        #
        # error = reward + (1 - done) * self.gamma * next_value - value
        #
        # error = error.item()

        self.buffer.add(state, action, reward, next_state, done)

        if len(self.buffer) > self.buffer.sample_size + 100:
            self._learn()

    def _learn(self):
        states, actions, rewards, next_states, dones = self.buffer.sample()
        targets = rewards + (1 - dones) * self.gamma * self.critic_target(next_states, self.actor_target(next_states)).detach()
        critic_loss = F.mse_loss(self.critic_main(states, actions), targets)
        actor_loss = -self.critic_main(states, self.actor_main(states)).mean()
        # update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        # update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        # update the target networks
        self._moving_average(self.actor_main, self.actor_target)
        self._moving_average(self.critic_main, self.critic_target)

    def _moving_average(self, main, target):
        for target_param, main_param in zip(target.parameters(), main.parameters()):
            target_param.data.copy_(self.tau * main_param.data + (1.0 - self.tau) * target_param.data)
