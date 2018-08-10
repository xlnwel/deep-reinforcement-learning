from agent import Agent
from monitor import interact
import gym
import numpy as np

env = gym.make('Taxi-v2')
alphas = [0.2, 0.3]
epsilons = [0.2, 0.3]
gammas = [0.4, 0.3]
best_val = 0
best_param = None
for alpha in alphas:
    for gamma in gammas:
        for eps in epsilons:
            for update_rule in ['Q_learning']:
                print("Alpha: {}".format(alpha))
                print("Gamma: {}".format(gamma))
                print("Epsilon: {}".format(eps))
                print("Update rule: {}".format(update_rule))
                agent = Agent(alpha, gamma, eps, update_rule=update_rule)
                avg_rewards, best_avg_reward = interact(env, agent, num_episodes=20000)
                if best_avg_reward > best_val:
                    best_param = (alpha, gamma, eps)
                    best_val = best_avg_reward

alpha, gamma, eps = best_param
print("Best alpha: {}".format(alpha))
print("Best gamma: {}".format(gamma))
print("Best epsilon: {}".format(eps))
print("Best Value: {}".format(best_val))