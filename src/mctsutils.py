from collections import defaultdict, deque
from typing import List, Tuple

import numpy as np

from aiutils import update_mean, update_var
from card import Card
from mcts import Node


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


def get_level_branching_factors(root: Node) -> List[Tuple[int, int]]:
    '''Return a list of (level, branching factor) for each node in the tree'''
    Q = deque([root])
    res = []

    # bfs
    level = 0
    while Q:
        N = len(Q)
        for i in range(N):
            n: Node = Q.popleft()
            x = len(n.children)
            if x > 0:
                res.append((level, x))
            Q = Q + deque(list(filter(lambda x: not x.is_leaf(), n.children)))
        level += 1
    return res


def get_level_visits(root: Node) -> List[Tuple[int, int]]:
    ''' Return a list of (level, visits) for each node in the tree'''
    Q = deque([root])
    res = []

    # bfs
    level = 0
    while Q:
        N = len(Q)
        for i in range(N):
            n: Node = Q.popleft()
            res.append((level, n.n))
            Q = Q + deque(list(filter(lambda x: not x.is_leaf(), n.children)))
        level += 1
    return res


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


def get_best_paths(root: Node, depth: int, num_paths: int) -> List[Node]:
    visited = defaultdict(bool)
    paths = []

    def get_best_paths_helper(node: Node, level: int, curr_path: List[Node]):
        curr_path.append(node)
        if level == depth:
            visited[node] = True
            paths.append(curr_path.copy())
        else:
            while True and len(paths) < num_paths:
                next_nodes = list(filter(lambda x: not visited[x] and x.n > 0, node.children))
                if not next_nodes:
                    paths.append(curr_path.copy())
                    visited[node] = True
                    curr_path.pop()
                    return
                scores = np.array([n.avg_value() for n in next_nodes])
                next_node_idx = np.argmax(scores)
                next_node = next_nodes[next_node_idx]
                get_best_paths_helper(next_node, level + 1, curr_path)
        curr_path.pop()

    get_best_paths_helper(root, 0, [])
    return paths


def get_buy_sequence(path: List[Node]) -> List[Card]:
    '''Given a path, return the associated list of card buys.'''
    return [(n.card, n.avg_value(), n.n) for n in path]


def get_depth(root: Node):
    Q = deque(root.children)
    depth = 0
    while Q:
        level_size = len(Q)
        for i in range(level_size):
            node = Q.popleft()
            Q = Q + deque(list(filter(lambda x: x and x.n > 0, node.children)))
        depth += 1

    return depth
