import pickle
from collections import Counter
from typing import Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression


def load(checkpoint: str, **kwargs):
    '''Load and return a saved object'''
    node = None
    try:
        node = pickle.load(open(checkpoint, 'rb'))
        return node
    except pickle.UnpicklingError:
        pass

    try:
        node = torch.load(checkpoint, **kwargs)
        return node
    except ImportError:
        pass

    raise ImportError(f'Failed to load model at {checkpoint} via torch and pickle.')


def save(file: str, obj):
    '''Save obj to pickled file'''
    with open(file, 'wb') as output:
        pickle.dump(obj, output, 4)


def softmax(x, t=1):
    '''Compute softmax values for each sets of scores in x.'''
    e_x = np.exp(x / t - np.max(x / t))
    return e_x / e_x.sum(axis=0)


def update_mean(n: int, prev_mean: float, x: float):
    '''Incremental update mean'''
    return (n - 1) / n * prev_mean + x / n


def update_var(n: int, prev_var: float, prev_mean: float, x: float):
    '''Incremental update variance'''
    if n == 1:
        return 0
    else:
        return (n - 2) / (n - 1) * prev_var + 1 / n * (x - prev_mean) ** 2


def classification_rate(labels: Iterable[int], sort=True, reverse=True):
    if sort:
        counts = Counter(sorted(labels, reverse=reverse))
    else:
        counts = Counter(labels)
    n = len(labels)
    return np.array(list(counts.keys())), np.array(list(counts.values())) / n


def copy_logistic_model(model: LogisticRegression, max_iter=10e5, penalty='none') -> LogisticRegression:
    copied_model = LogisticRegression(max_iter=max_iter, penalty=penalty)
    copied_model.coef_ = model.coef_.copy()
    copied_model.classes_ = model.classes_.copy()
    copied_model.intercept_ = model.intercept_
    return copied_model


def plot_card_counts_stacked(decks: List[Counter], limit=None, skip=1, trim=None):
    '''Produce a stacked plot of card counts every skip number of iterations,
        including only cards in limit'''
    n = len(decks)
    counts = dict()
    for i, d in enumerate(decks):
        if trim and i >= trim:
            break
        for k, v in d.items():
            if limit and k not in limit:
                continue
            if k not in counts.keys():
                counts[k] = [0] * (trim if trim else (n // skip + 1))
            counts[k][(i // skip)] = v

    df = pd.DataFrame(counts)
    df.plot.area()
    plt.xlabel('Iterations')
    plt.ylabel('Card count')
    plt.show()


def plot_card_counts(decks: List[Counter], limit=None, skip=1, trim=None):
    '''Produce a line plot of card counts every skip number of iterations,
    including only cards in limit'''
    n = len(decks)
    counts = dict()
    for i, d in enumerate(decks):
        if trim and i >= trim:
            break
        if i % skip != 0:
            continue
        for k, v in d.items():
            if limit and k not in limit:
                continue
            if k not in counts.keys():
                counts[k] = [0] * (trim if trim else (n // skip + 1))
            counts[k][(i // skip)] = v

    legend = []
    for k, v in counts.items():
        plt.plot(v)
        legend.append(k)
    plt.legend(legend)
    plt.xlabel('Iterations')
    plt.ylabel('Card count')
    plt.show()


def plot_card_counts_scatter(decks: List[Counter], limit=None, skip=1):
    '''Produce a scatter plot of card counts every skip number of iterations,
    including only cards in limit'''
    n = len(decks)
    counts = dict()
    for i, d in enumerate(decks):
        if i % skip != 0:
            continue
        for k, v in d.items():
            if limit and k not in limit:
                continue
            if k not in counts.keys():
                counts[k] = [0] * (n // skip)
            counts[k][(i // skip)] = v

    legend = []
    for k, v in counts.items():
        plt.scatter(range(n // skip), v)
        legend.append(k)
    plt.legend(legend)
    plt.xlabel('Iterations')
    plt.ylabel('Card count')
    plt.show()


def plot_scores(score: np.array):
    '''Produce a line plot of final score every skip number of iterations'''
    avgs = []
    score_sum = 0
    for i, x in enumerate(score):
        score_sum += score[i]
        avgs.append(score_sum / (i + 1))
    plt.scatter(range(len(score)), score, s=0.9)
    plt.plot(avgs, 'r')
    plt.xlabel('Iterations')
    plt.ylabel('Score')
    plt.show()
