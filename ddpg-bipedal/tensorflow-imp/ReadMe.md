There's a problem with this code that the trainable variables in actor doesn't receive any gradients.

The code should be able to run by `python3 train.py` if the required packages(such as `gym` and `box2d`) are installed