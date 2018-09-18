import gym
import numpy as np
from RL.openai.cartpole.policy import basic_policy
from RL.openai.cartpole.base import RLBase


if __name__ == "__main__":
    env = gym.make("CartPole-v0")
    playground = RLBase(1000, 500, env)
    playground.trial(basic_policy)

    print(np.mean(playground.totals))
    print(np.std(playground.totals))
    print(np.min(playground.totals))
    print(np.max(playground.totals))
