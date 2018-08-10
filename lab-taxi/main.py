from agent import Agent
from monitor import interact
import gym
import numpy as np

env = gym.make('Taxi-v2')
alphas = [0.8, 1]
cs = [1e5]
gammas = [0.1, 0.5]
for alpha in alphas:
    for gamma in gammas:
        for c in cs:
            for update_rule in ['Q_learning']:
                print("Alpha: {}".format(alpha))
                print("Gamma: {}".format(gamma))
                print("C: {}".format(c))
                print("Update rule: {}".format(update_rule))
                agent = Agent(alpha, gamma, c, update_rule=update_rule)
                avg_rewards, best_avg_reward = interact(env, agent, num_episodes=20000)