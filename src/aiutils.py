import pickle
from mcts import Node
from card import Card
from collections import Counter, deque
from typing import List
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def load(checkpoint: str):
    '''Load and return a saved object'''
    node = pickle.load(open(checkpoint, 'rb'))
    return node

def save(file: str, obj):
    '''Save obj to pickled file'''
    with open (file, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

def print_path(path: List[Node]):
    print('-->'.join([str(node.card) for node in path]))


def path_helper(curr: Node, acc: List[Node], key):
    if curr.n > 0 and curr.children:
        child = max(curr.children, key=key)
        acc.append(child)
        path_helper(child, acc)

def best_path(root: Node) -> List[Node]:
    '''Return the max-valued path from root to leaf'''
    path = [root]
    path_helper(root, path, lambda x: x.v)
    return path

def get_branching_factors(root: Node) -> List[int]:
    Q = deque(root.children)
    L = []
    while Q:
        N = len(Q)
        for i in range(N):
            n = Q.popleft()
            L.append(sum(1 if not n.is_leaf() > 0 else 0 for c in n.children))
            Q = Q + deque(list(filter(lambda x: not x.is_leaf(), n.children)))

    return L


def get_path(root: Node, leaf: Node):
    path = []
    curr = leaf
    while curr and curr.parent != root:
        path.append(curr)
        curr = curr.parent
    path.reverse()
    return path

def get_path_lengths(root: Node) -> List[int]:
    lengths = []
    def get_path_lengths_helper(curr: Node, acc: int):
        if curr.is_leaf():
            lengths.append(acc)
        for c in curr.children:
            get_path_lengths_helper(c, acc+1)
    get_path_lengths_helper(root, 0)
    return lengths

def get_most_visited_paths_at_depth(root: Node, k: int, p: int):
    '''Return the p most traversed length-k path from starting node (not virtual root).'''
    assert k > 0, 'k must be positive'
    assert p > 0, 'p must be positive'

    Q = deque(root.children)
    # bfs from start nodes
    for i in range(1, k+1):
        l = len(Q)
        print(len(Q))
        for j in range(l):
            n = Q.popleft()
            Q = Q + deque(list(filter(lambda x: x and x.n > 0, n.children)))

    paths = sorted(Q, key=lambda x: x.n, reverse=True)
    print(len(paths))
    for i, n in enumerate(paths):
        paths[i] = get_path(root, n)

    return paths[:p]

def get_most_visited_leaf(root: Node) -> Node:
    '''Returns the leaf with the highest visit count'''
    max_n = 0
    max_leaf = None
    def get_most_visited_leaf_helper(curr: Node):
        nonlocal max_n
        nonlocal max_leaf
        if not curr.children or max([c.n for c in curr.children]) == 0:
            if curr.n > max_n:
                max_n = curr.n
                max_leaf = curr
        else:
            for child in curr.children:
                get_most_visited_leaf_helper(child)
    get_most_visited_leaf_helper(root)
    return max_leaf

def most_visited_path(root: Node) -> List[Node]:
    '''Return the most-visited path from root to leaf'''
    curr = get_most_visited_leaf(root)
    return get_path(root, curr)

def get_buy_sequence(path: List[Node]) -> List[Card]:
    '''Given a path, return the associated list of card buys.'''
    return [n.card for n in path]

def get_card_counts(cards: List[Card]) -> Counter:
    return Counter(str(c) for c in cards)

def plot_stacked_card_counts(decks: List[Counter], limit=None, skip=1):
    '''Produce a stacked plot of card counts every skip number of iterations,
        including only cards in limit'''
    n = len(decks)
    counts = dict()
    for i, d in enumerate(decks):
        for k, v in d.items():
            if limit and k not in limit:
                continue
            if k not in counts.keys():
                counts[k] = [0] * (n // skip)
            counts[k][(i // skip)] = v

    df = pd.DataFrame(counts)
    df.plot.area()
    plt.xlabel('Iterations')
    plt.ylabel('Card count')
    plt.show()

def plot_card_counts(decks: List[Counter], limit=None, skip=1):
    '''Produce a line plot of card counts every skip number of iterations,
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
    plt.scatter(range(len(score)), score)
    plt.plot(avgs, 'r')
    plt.xlabel('Iterations')
    plt.ylabel('Score')
    plt.show()


