[//]: # "Image References"

[image1]: https://user-images.githubusercontent.com/10624937/42135610-c37e0292-7d12-11e8-8228-4d3585f8c026.gif "Trained Agent"

# Notes

Adding batch normalization layer impair performance.

Inserting action to the second layer of critic significantly outperforms adding it to the first. 

Adding noise to parameters (with noise decay) also improves performance a lot (achieving average score -150 within only 200 episodes, significantly better than the baseline, which achieves -500 using 1000 episodes).

Here's the thing, it seems denoising parameters after action is chosen somewhat impairs the performance a bit. No idea about the reason. 

# Actor-Critic Methods

### Instructions

Open `DDPG.ipynb` to see an implementation of DDPG with OpenAI Gym's Pendulum environment.

### Results

![Trained Agent][image1]