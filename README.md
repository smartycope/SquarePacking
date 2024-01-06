# SquarePacking

This attempts to use a reinforcement learning algorithm based on a [Deep Deterministic Policy Gradient](https://spinningup.openai.com/en/latest/algorithms/ddpg.html) model for use with a continuous observation *and* action space, in order to solve the [*Square Packing in a Square*](https://en.wikipedia.org/wiki/Square_packing) problem for N=11 squares.

That was a very intelligent sounding sentence. Let's break it down:
- The [*Square Packing in a Square*](https://en.wikipedia.org/wiki/Square_packing) problem is a problem in mathematics where the goal is to pack `N` squares with a side length of 1 into another square, while wasting as little space as possible. See the [Wikipedia page](https://en.wikipedia.org/wiki/Square_packing) for more details.
    - There are known configurations for N=1-10 squares, but 11 (and some others) are only approximately solved. This tries to find a more optimal configuration for N=11 squares by using RL instead of pure math.
- A [DDPG](https://spinningup.openai.com/en/latest/algorithms/ddpg.html) model is a kind of [actor-critic](https://www.mathworks.com/help/reinforcement-learning/ref/rl.agent.rlacagent.html) setup (not technically a model) that allows continuous rather than discrete observation and action spaces. This is important, because I want to find a very precise solution, as opposed to infinitely increasing the discrete resolution of steps the AI can take.


## Running
I have a requirements.txt for completeness sake, but you can clone the repo and run all the cells in [Attempt3.ipynb](Attempt3.ipynb), and it should install them for you and then start running.
