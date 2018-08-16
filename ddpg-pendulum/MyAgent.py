import numpy as np
import torch
import torch.nn.functional as F
from MyModel import *
import copy
from collections import deque, namedtuple
import random

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class ReplayBuffer():
    def __init__(self, sample_size, max_len=int(1e5)):
        self.buffer = deque(maxlen=max_len)
        self.sample_size = sample_size
        self.experience = namedtuple('experience', ('state', 'action', 'reward', 'next_state', 'done'))

    def add(self, state, action, reward, next_state, done):
        exp = self.experience(state, action, reward, next_state, done)
        self.buffer.append(exp)

    def sample(self):
        exps = random.sample(self.buffer, self.sample_size)

        states = torch.from_numpy(np.vstack([e.state for e in exps])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in exps])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in exps])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in exps])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in exps]).astype(np.uint8)).float().to(device)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


class Agent():
    def __init__(self, state_size, action_size, batch_size=64, actor_alpha=1e-4, critic_alpha=1e-3, gamma=0.99, tau=1e-3):
        # hyperparameters
        self.gamma = gamma
        self.tau = tau
        # env info
        self.state_size = state_size
        self.action_size = action_size
        # replay buffer
        self.buffer = ReplayBuffer(sample_size=batch_size)
        # actor-critic
        self.actor_main = Actor(state_size, action_size).to(device)
        self.critic_main = Critic(state_size, action_size).to(device)
        # target actor-critic
        self.actor_target = copy.deepcopy(self.actor_main)
        self.critic_target = copy.deepcopy(self.critic_main)
        # optimizer
        self.actor_optimizer = torch.optim.Adam(self.actor_main.parameters(), lr=actor_alpha)
        self.critic_optimizer = torch.optim.Adam(self.critic_main.parameters(), lr=critic_alpha)
        # Noise process
        self.noise = OUNoise(action_size)

    def act(self, state):
        state = state.reshape((1, -1))
        state = torch.from_numpy(state).float().to(device)

        self.actor_main.eval()
        with torch.no_grad():
           action = self.actor_main(state).cpu().numpy()
        self.actor_main.train()
        action = self.noise.sample()

        return np.clip(action, -2, 2)

    def step(self, state, action, reward, next_state, done):
        state = state.reshape((1, -1))
        next_state = next_state.reshape((1, -1))
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
        self._moving_average()

    def _moving_average(self):
        for actor_target_param, actor_main_param in zip(self.actor_target.parameters(), self.actor_main.parameters()):
            actor_target_param.data.copy_(self.tau * actor_main_param.data + (1.0 - self.tau) * actor_target_param.data)
        for critic_target_param, critic_main_param in zip(self.critic_target.parameters(), self.critic_main.parameters()):
            critic_target_param.data.copy_(self.tau * critic_main_param.data + (1.0 - self.tau) * critic_target_param.data)


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state