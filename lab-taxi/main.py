from agent import Agent
from monitor import interact
import gym
import numpy as np

env = gym.make('Taxi-v2')
iterations = 10
alphas = np.random.uniform(0.1, size=iterations)
gammas = np.random.uniform(0.1, size=iterations)
epsilons = np.random.uniform(0.1, 0.4, size=iterations)
records = {}
for alpha, gamma, eps in zip(alphas, gammas, epsilons):
    for update_rule in ['Q_learning']:
        print("Alpha: {}".format(alpha))
        print("Gamma: {}".format(gamma))
        print("Epsilon: {}".format(eps))
        print("Update rule: {}".format(update_rule))
        agent = Agent(alpha, gamma, eps, update_rule=update_rule)
        avg_rewards, best_avg_reward = interact(env, agent, num_episodes=20000)
        records[(alpha, gamma, eps)] = best_avg_reward
print('--------------------------------------------------')
params = sorted(records.keys(), key=records.get)
for alpha, gamma, eps in params:
    print()
    print("Best alpha: {}".format(alpha))
    print("Best gamma: {}".format(gamma))
    print("Best epsilon: {}".format(eps))
    print("Best average value: {}".format(records[(alpha, gamma, eps)]))