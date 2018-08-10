from agent import Agent
from monitor import interact
import gym
import numpy as np

env = gym.make('Taxi-v2')
alphas = [0.2, 0.3]
epsilons = [0.2, 0.3]
gammas = [0.4, 0.3]
records = {}
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
                records[(alpha, gamma, eps)] = best_avg_reward
print('--------------------------------------------------')
params = sorted(records.keys(), records.get())
for alpha, gamma, eps in params:
    print("Best alpha: {}".format(alpha))
    print("Best gamma: {}".format(gamma))
    print("Best epsilon: {}".format(eps))
    print("Best average value: {}".format(records[(alpha, gamma, eps)]))