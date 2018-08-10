import numpy as np
from collections import defaultdict


class Agent:

    def __init__(self, alpha = 1e-2, gamma=0.1, lambdA=1, epsilon=1e5, nA=6, update_rule='Q_learning'):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.alpha = alpha
        self.gamma = gamma

        self.epsilon = epsilon
        # time tracker
        self.t = 0
        # Q learning or expected sarsa
        self.update_rule = update_rule
        # parameters for eligibility trace
        self.E = defaultdict(lambda: np.zeros(self.nA))
        self.lambdA = lambdA
        # next action
        self.action = None

    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        return self.action if self.action else self.epsilon_greedy(state)

    def epsilon_greedy(self, state):
        return np.random.choice(np.arange(self.nA)) if np.random.uniform() < self.epsilon else np.argmax(self.Q[state])

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
        next_action = None
        if done:
            delta = reward - self.Q[state][action]
        else:
            next_action = self.epsilon_greedy(next_state)
            delta = reward + self.Q[next_state][next_action] - self.Q[state][action]
        self.E[state][action] += 1
        for s in self.E.keys():
            for a in range(self.nA):
                self.Q[s][a] += self.alpha * delta * self.E[s][a]
                self.E[s][a] *= self.lambdA * self.gamma
        self.action = next_action
        self.t += 1
        if self.t % 1e4 == 0:
            self.epsilon -= 0.01
