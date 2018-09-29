[//]: # "Image References"

[image1]: https://user-images.githubusercontent.com/10624937/42135608-be87357e-7d12-11e8-8eca-e6d5fabdba6b.gif "Trained Agent"

# Notes for Pytorch version

The agent finally achieve mean rewards more than 30 over 100 episodes at last, just the time I was about the to give up ;)

The sad part is it suddenly behaved badly at the end of the last 100 episodes, otherwise, it could achieve more than 60 :(



I tried to improve it further by adding batch normalization, using priority queue instead of deque, adding noise decay. Nothing seems promising. 

I don't know where the error is: I save the model and reload it, then the agent seems start all over again. Maybe it has something to do with replay buffer being cleared after restart training, maybe.

# Actor-Critic Methods

### Instructions

Open `DDPG.ipynb` to see an implementation of DDPG with OpenAI Gym's BipedalWalker environment.

### Results

![Trained Agent][image1]
