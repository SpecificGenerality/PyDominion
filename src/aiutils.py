import pickle
from collections import Counter, deque
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from card import Card
from mcts import Node


def load(checkpoint: str):
    '''Load and return a saved object'''
    node = pickle.load(open(checkpoint, 'rb'))
    return node


def save(file: str, obj):
    '''Save obj to pickled file'''
    with open(file, 'wb') as output:
        pickle.dump(obj, output, 4)


def print_path(path: List[Node]):
    print('-->'.join([str(node.card) for node in path]))


def path_helper(curr: Node, acc: List[Node], key):
    if curr.n > 0 and curr.children:
        child = max(curr.children, key=key)
        acc.append(child)
        path_helper(child, acc, key=key)


def best_path(root: Node) -> List[Node]:
    '''Return the max-valued path from root to leaf'''
    path = [root]
    path_helper(root, path, lambda x: x.v)
    return path


def update_mean(n: int, prev_mean: float, x: float):
    '''Incremental update mean'''
    return (n - 1) / n * prev_mean + x / n


def update_var(n: int, prev_var: float, prev_mean: float, x: float):
    '''Incremental update variance'''
    if n == 1:
        return 0
    else:
        return (n - 2) / (n - 1) * prev_var + 1 / n * (x - prev_mean) ** 2


def get_branching_factor_stats(root: Node) -> List[int]:
    '''Calculate the mean and variance of tree branching factor'''
    Q = deque(root.children)
    mean, var = 0, 0
    # bfs
    while Q:
        k = 1
        N = len(Q)
        for i in range(N):
            n: Node = Q.popleft()
            x = sum(1 if not n.is_leaf() else 0 for c in n.children)
            prev_mean = mean
            mean = update_mean(k, prev_mean, x)
            var = update_var(k, var, prev_mean, x)
            Q = Q + deque(list(filter(lambda x: not x.is_leaf(), n.children)))
            k += 1
    return mean, var


def get_path(root: Node, leaf: Node):
    '''Get the path from leaf to root'''
    path = []
    curr = leaf
    while curr and curr != root:
        path.append(curr)
        curr = curr.parent

    if curr == root:
        path.append(root)

    path.reverse()
    return path


def get_most_visited_paths_at_depth(root: Node, k: int, p: int):
    '''Return the p most traversed length-k path from game start nodes (not virtual root).'''
    Q = deque(root.children)
    # find the level-k nodes via bfs
    for i in range(1, k + 1):
        level_length = len(Q)
        for j in range(level_length):
            n = Q.popleft()
            Q = Q + deque(list(filter(lambda x: x and x.n > 0, n.children)))

    paths = sorted(Q, key=lambda x: x.n, reverse=True)
    for i, n in enumerate(paths):
        paths[i] = get_path(root, n)[1:]

    return paths[:p]


def get_buy_sequence(path: List[Node]) -> List[Card]:
    '''Given a path, return the associated list of card buys.'''
    return [n.card for n in path]


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
