import tensorflow as tf
from RL.openai.cartpole.util import discount_rewards, discount_and_normalize_rewards


class RLBase:
    def __init__(self, n_iter, n_episode, env):
        self.n_iter = n_iter
        self.n_episode = n_episode
        self.env = env
        self.totals = []

    def trial(self, policy):
        for episode in range(self.n_episode):
            episode_rewards = 0
            obs = self.env.reset()
            for step in range(self.n_iter):
                action = policy(obs)
                obs, reward, done, info = self.env.step(action)
                episode_rewards += reward
                if done:
                    break
            self.totals.append(episode_rewards)


class RLNNBase:
    def __init__(self, n_iter, n_episode, n_game_per_update, save_iter, learning_rate, env):
        self.n_iter = n_iter
        self.n_episode = n_episode
        self.n_game_per_update = n_game_per_update
        self.save_iter = save_iter
        self.learning_rate = learning_rate
        self.env = env

