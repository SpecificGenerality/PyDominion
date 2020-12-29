import sys
from typing import Callable, Iterable, List

import numpy as np

from card import Card
from cursecard import Curse


class Node:
    def __init__(self, parent=None, card=None, n=0, v=0):
        # TODO: For non-sandbox, maybe states should be stored?
        # self.state = s
        self.parent = parent
        # the action := the card played to get from parent to current
        self.card = card
        # number of times visited
        self.n = n
        # node value
        self.v = v
        self.children = []

    # UCB1 formula
    def score(self, C):
        return self.v / self.n + C * np.sqrt(np.log(self.parent.n) / self.n) if self.n > 0 else sys.maxsize

    def update_v(self, f: Callable[[Iterable], float]):
        vals = np.array([n.v for n in self.children if n.n > 0])
        self.v = f(vals)

    def update(self, delta: int):
        self.v += delta
        self.n += 1

    def backpropagate(self, delta: int):
        self.update(delta)

        if self.parent == self:
            return

        self.parent.backpropagate(-delta)
        return

    def add_unique_children(self, cards: List[Card]):
        for c in cards:
            found = False
            if isinstance(c, Curse):
                continue
            for child in self.children:
                if str(c) == str(child.card):
                    found = True
            if not found:
                self.children.append(Node(parent=self, card=c))

    def get_child_node(self, card: Card):
        for child in self.children:
            if str(child.card) == str(card):
                return child

    def is_leaf(self) -> bool:
        if not self.children:
            return True
        for c in self.children:
            if c.n > 0:
                return False
        return True

    # size of the subtree rooted at the current node
    def size(self):
        acc = 1
        for child in self.children:
            if not child.is_leaf():
                acc += child.size()
        return acc

    def __str__(self):
        return f'{self.parent.card}<--n: {self.n}, v: {self.v}, c: {self.card}-->{[str(c.card) for c in self.children]}\n'

    def __repr__(self):
        return str(self)
