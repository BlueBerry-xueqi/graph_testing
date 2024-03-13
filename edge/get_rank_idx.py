import numpy as np
import random
from scipy.special import softmax
from scipy.stats import entropy


def Random_rank_idx(x):
    random_rank_idx = random.sample(range(0, len(x)), len(x))
    return random_rank_idx


def DeepGini_rank_idx(x):
    gini_score = 1 - np.sum(np.power(x, 2), axis=1)
    gini_rank_idx = gini_score.argsort()[::-1]
    return gini_rank_idx


def LC_rank_idx(x):
    value = 1 - x.max(1)
    rank_idx = np.argsort(value)[::-1]
    return rank_idx


def LC_variant_rank_idx(x):
    value = x.min(1)
    rank_idx = np.argsort(value)[::-1]
    return rank_idx


def Entropy_rank_idx(x):
    prob_dist = np.array([i / np.sum(i) for i in x])
    entropy_res = entropy(prob_dist, axis=1)
    entropy_rank_idx = np.argsort(entropy_res)[::-1]
    return entropy_rank_idx


def Margin_rank_idx(x):
    output_sort = np.sort(x)
    margin_score = output_sort[:, -1] - output_sort[:, -2]
    rank_idx = np.argsort(margin_score)
    return rank_idx


def Variance_rank_idx(x):
    variance_score = np.var(x, axis=1)
    rank_idx = np.argsort(variance_score)
    return rank_idx




