import numpy as np
from collections import defaultdict


class Agent:

    def __init__(self, alpha = 1e-2, gamma=0.1, epsilon=1e5, nA=6, update_rule='Q_learning'):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        # model 1 with 9.48 best average rewards
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.alpha = alpha
        self.gamma = gamma

        self.epsilon = epsilon
        # time tracker
        self.t = 0
        # Q learning or expected sarsa
        self.update_rule = update_rule

    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        epsilon_policy = lambda: np.random.choice(np.arange(self.nA)) if np.random.uniform() < self.epsilon else \
            np.argmax(self.Q[state])
        return epsilon_policy()

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        delta = reward - self.Q[state][action] + self.gamma * (0 if done else np.max(self.Q[next_state]))
        self.Q[state][action] += self.alpha * delta
        self.t += 1
        if self.t % 1e4 == 0:
            self.epsilon -= 0.1
