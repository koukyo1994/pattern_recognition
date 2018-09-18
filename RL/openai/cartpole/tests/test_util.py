import numpy as np
import numpy.testing as npt

from RL.openai.cartpole.util import discount_rewards, discount_and_normalize_rewards


def test_discount_rewards():
    rewards = [10, 0, -50]
    discount_rate = 0.8
    npt.assert_almost_equal(discount_rewards(rewards, discount_rate),
                            np.array([-22., -40., -50.]),
                            decimal=8)


def test_discount_and_normalize_rewards():
    all_rewards = [[10, 0, -50], [10, 20]]
    discount_rate = 0.8
    return_rewards = discount_and_normalize_rewards(all_rewards, discount_rate)
    npt.assert_almost_equal(return_rewards[0],
                            np.array([-0.28435071, -0.86597718, -1.18910299]),
                            decimal=8)
    npt.assert_almost_equal(return_rewards[1],
                            np.array([1.26665318, 1.0727777]),
                            decimal=8)


if __name__ == "__main__":
    test_discount_rewards()
    test_discount_and_normalize_rewards()
