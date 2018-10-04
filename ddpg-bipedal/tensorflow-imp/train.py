import gym
import random
import tensorflow
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import utils.utils as utils
import tensorflow as tf
from ddpg_tf import DDPG

env = gym.make('BipedalWalker-v2')
env.seed(0)

sess = tf.Session()
agent = DDPG('ddpg', utils.load_args(), sess=sess)
agent.restore()
def ddpg(n_episodes=10000, max_t=1000):
    scores_deque = deque(maxlen=100)
    scores = []
    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        score = 0
        for t in range(max_t):
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break 
        scores_deque.append(score)
        scores.append(score)
        average_score = np.mean(scores_deque)
        print('\rEpisode {}\tAverage Score: {:.2f}\tScore: {:.2f}'.format(i_episode, average_score, score), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, average_score))
            agent.save()
    return scores

scores = ddpg()

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(scores)+1), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()