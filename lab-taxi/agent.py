import numpy as np
from collections import defaultdict

class Agent:

    def __init__(self, alpha = 1e-2, gamma=0.1, c = 1e5, nA=6, update_rule='Q_learning'):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.alpha = alpha
        self.gamma = gamma
        # variables for UCB
        self.c = c
        self.N = defaultdict(lambda: np.zeros(self.nA))
        self.t = 0
        # number of episodes is passed
        self.n_eps = 1
        self.c_decay = False
        self.alpha_decay = False
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
        ucb = lambda a: a if self.N[state][a] == 0 else \
            self.Q[state][a] + self.c * np.sqrt(np.log(self.t) / self.N[state][a])
        return np.argmax([ucb(a) for a in range(self.nA)])

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
        expected_sarsa = lambda s: self.gamma * np.average(self.Q[s],
                                                           weights=self.Q[s] if self.Q[s].any() else None)
        Q_learning = lambda s: self.gamma * np.max(self.Q[s])
        target_value = reward if done else \
            reward + (Q_learning(next_state) if self.update_rule == 'Q_learning' else expected_sarsa(next_state))
        self.Q[state][action] = self.Q[state][action] + self.alpha * (target_value - self.Q[state][action])
        self.N[state][action] += 1
        self.t += 1
        if done:
            self.n_eps += 1
            self.c_decay = True
            self.alpha_decay = True

        if self.c_decay and self.t % 2e4 == 0:
            self.c /= 5
            self.c_decay = False

        # if self.alpha_decay and self.n_eps % 1000 == 0:
        #     self.alpha = max(self.alpha/2, 0.1)
        #     self.alpha = False
