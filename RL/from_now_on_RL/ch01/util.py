import numpy as np


class Greedy:
    def __init__(self, num_lever, seed=None):
        self.num_lever = num_lever
        self.reset()
        self.num_search_step = 100
        self.seed = seed

    def reset(self):
        self.num_play = np.array([0. for i in range(self.num_lever)])
        self.reward = np.array([0. for i in range(self.num_lever)])

    def set_num_search(self, num_search_step):
        self.num_search_step = num_search_step

    def set_seed(self, seed):
        self.seed = seed

    def set_reward(self, reward, idx):
        num_play = np.zeros(reward.shape)
        num_play[idx] = 1.0
        self.num_play + num_play
        self.reward + reward

    def is_fully_searched(self):
        return np.where(self.num_play > self.num_search_step)[0]

    def step(self):
        if self.seed:
            np.random.seed(self.seed)
        not_yet_arrays = self.is_fully_searched()
        if not_yet_arrays.size > 0:
            return np.random.choice(not_yet_arrays, 1)[0]

        return (self.reward / (self.num_play + 1.0)).argmax()


class EpsilonGreedy(Greedy):
    def __init__(self, num_lever, epsilon=0.1, seed=None):
        super().__init__(num_lever, seed)
        self.epsilon = epsilon

    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

    def is_random_choice(self):
        return np.random.binomial(1, self.epsilon)

    def step(self):
        if self.seed:
            np.random.seed(self.seed)
        not_yet_arrays = self.is_fully_searched()
        if not_yet_arrays.size > 0:
            return np.random.choice(not_yet_arrays, 1)[0]

        if self.is_random_choice():
            return np.random.choice(np.arange(self.num_lever), 1)[0]
        return (self.reward / (self.num_play + 1.0)).argmax()
