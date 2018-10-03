import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from util import Greedy, EpsilonGreedy


class Env:
    def __init__(self, levers):
        self.num_lever = levers.size
        self.levers = levers

    def set_new_levers(self, levers):
        self.levers = levers

    def get_reward(self, idx):
        result = np.random.binomial(1, self.levers, self.levers.shape)
        mask = np.zeros(self.levers.shape)
        mask[idx] = 1
        return result * mask


if __name__ == '__main__':
    levers = np.array([0.2, 0.3, 0.4, 0.5])
    env = Env(levers)
    greedy = Greedy(levers.size, seed=101)
    eps_greedy = EpsilonGreedy(levers.size, epsilon=0.1, seed=101)

    num_trial = 10000
    num_step = 10000
    greedy_average_rewards = np.zeros((num_trial, ))
    eps_greedy_average_rewards = np.zeros_like(greedy_average_rewards)
    greedy_sum_average_rewards = 0
    eps_greedy_sum_average_rewards = 0
    trial_cnt = 0
    for i in tqdm(range(num_trial)):
        greedy_reward_in_trial = 0
        eps_greedy_reward_in_trial = 0
        for i in range(num_step):
            action = greedy.step()
            reward = env.get_reward(action)
            greedy.set_reward(reward, action)
            greedy_reward_in_trial += reward.sum()

            action = eps_greedy.step()
            reward = env.get_reward(action)
            eps_greedy.set_reward(reward, action)
            eps_greedy_reward_in_trial += reward.sum()

        greedy.reset()
        eps_greedy.reset()
        greedy_sum_average_rewards += greedy_reward_in_trial / 10000.0
        eps_greedy_sum_average_rewards += eps_greedy_reward_in_trial / 10000.0
        trial_cnt += 1

        greedy_average_rewards[i] = greedy_sum_average_rewards / trial_cnt
        eps_greedy_average_rewards[i] = \
            eps_greedy_sum_average_rewards / trial_cnt

    t = np.arange(0, 10000)
    plt.plot(t, greedy_average_rewards, label="Greedy")
    plt.plot(t, eps_greedy_average_rewards, label=r"$\epsilon$-greedy")
    plt.xlabel("The number of trial")
    plt.ylabel("Average reward")
    plt.legend()
    plt.show()
