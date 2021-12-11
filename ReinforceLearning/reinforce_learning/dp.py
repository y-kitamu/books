"""Dynamic Programingによって最適な方策を探索する
本のP.105の問題の最適方策の探索
"""
import math

import numpy as np


def poisson_dist(self, lambd, lower=0, upper=20):
    dists = np.zeros((upper - lower + 1, 1))
    for i in range(lower, upper + 1):
        dists[i] = lambd ** i * np.exp(-lambd) / math.factorial(i)
    return dists


class PolicyOptimizer:
    """
    """
    N_MAX_CARS = 20   # maximum number of cars at each office
    N_MAX_MOVE_CARS = 5  # maximum number of cars to move to the other office in one day
    LAMBDA_RETURN1 = 3
    LAMBDA_RETURN2 = 2
    LAMBDA_RENTAL1 = 3
    LAMBDA_RENTAL2 = 4

    def __init__(self, theta=0.5, gamma=0.9):
        self.value_function = np.zeros((self.N_MAX_CARS + 1, self.N_MAX_CARS + 1))
        self.policies = np.zeros(self.value_function.shape)
        self.return_probs1 = poisson_dist(self.LAMBDA_RETURN1)
        self.return_probs2 = poisson_dist(self.LAMBDA_RETURN2)
        self.rental_probs1 = poisson_dist(self.LAMBDA_RENTAL1)
        self.rental_probs2 = poisson_dist(self.LAMBDA_RENTAL2)
        self.trans_probs = self.calculate_transition_probability()
        self.theta = theta
        self.gamma = gamma

    def calculate_transition_probability(self):

    def policy_evaluation(self):
        for row in range(self.N_MAX_CARS + 1):
            for col in range(self.N_MAX_CARS + 1):
                value = self.value_function[row, col]

    def _update_value_function(self):
        for n1 in range(self.N_MAX_CARS + 1):
            for n2 in range(self.N_MAX_CARS + 1):
                # レンタカー移動 -> 返却 -> 貸出
                value = self.value_function[n1, n2]
                n_move = self.policies[n1, n2]
                s1 = n1 + n_move # レンタカー移動後のoffice1のレンタカー台数
                s2 = n2 - n_move
                self.value_function[n1, n2] = self._calc_reward(s1, s2)

    def _calc_reward(self, n_cars1, n_cars2):

        reward = 0
        return reward


    def greedy_policy_improvement(self):
        is_policy_stable = True
        return is_policy_stable

    def policy_iteration(self, n_iterations=10):
        for i in range(n_iterations):
            self.policy_evaluation()
            is_policy_stable = self.greedy_policy_improvement()
            if is_policy_stable:
                break
        return self.value_function, self.policy
