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
        self.Q1 = defaultdict(lambda: np.zeros(self.nA))
        self.alpha1 = 0.68
        self.gamma1 = 0.36
        # model 2 with 9.58 best average rewards
        self.Q2 = defaultdict(lambda: np.zeros(self.nA))
        self.alpha2 = 0.15
        self.gamma2 = 0.29

        self.epsilon = 0.33
        # time tracker
        self.t = 0
        # Q learning or expected sarsa
        self.update_rule = update_rule
        self.picked = False
        self.Q = lambda s: self.Q1[s] + self.Q2[s]

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
            np.argmax(self.Q2[state])
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
        delta1 = reward - self.Q1[state][action] + self.gamma1 * (0 if done else np.max(self.Q1[next_state]))
        self.Q1[state][action] += self.alpha1 * delta1
        delta2 = reward - self.Q2[state][action] + self.gamma2 * (0 if done else np.max(self.Q2[next_state]))
        self.Q2[state][action] += self.alpha1 * delta2
        self.t += 1
        if self.t % 1e4 == 0:
            self.epsilon -= 0.1
