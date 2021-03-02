import sys
from typing import Callable, Iterable, List, Tuple

import numpy as np

from aiutils import load
from card import Card
from cursecard import Curse
from enums import GameConstants, Zone
from state import State


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
        return self.v / self.n + 2 * C * np.sqrt(2 * np.log(self.parent.n) / self.n) if self.n > 0 else sys.maxsize

    def avg_value(self):
        return self.v / self.n if self.n > 0 else -sys.maxsize

    def update(self, delta: int):
        self.v += delta
        self.n += 1

    def backpropagate(self, rewards: Tuple, start_idx: int = 0):
        n = len(rewards)
        i = start_idx
        node = self
        while node.parent and node.parent != node:
            node.update(rewards[i])
            i = (i + 1) % n
            node = node.parent

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
        return f'Parent: {self.parent.card} << (n: {self.n}, v: {self.v}, Q: {self.avg_value():.3f} c: {self.card}) >> Children: {[(i, str(c.card), "%d" % c.n, "%.3f" % c.avg_value()) for i, c in enumerate(self.children)]}\n'

    def __repr__(self):
        return str(self)


class GameTree:
    def __init__(self, root: Node = Node(), train: bool = False, C: Callable[[int], float] = lambda x: max(1, min(25, 25 / np.sqrt(x)))):
        self._root: Node = root
        self._node: Node = root
        self.train: bool = train
        self._in_tree: bool = True
        self.C = C

        self._root.parent = self._root
        # To prevent clobbering trees loaded from file
        if not self._root.children:
            self._root.children = [Node(parent=self._root) for _ in range(GameConstants.StartingHands)]

    @classmethod
    def load(cls, path: str, train: bool):
        root = load(path)
        assert isinstance(root, Node)
        return cls(root, train)

    @property
    def node(self):
        return self._node

    @property
    def in_tree(self):
        return self._in_tree

    def reset(self, s: State):
        self._in_tree = True
        self._node = self._root.children[s.get_treasure_card_count(0, Zone.Hand) + s.get_treasure_card_count(0, Zone.Play) - 2]

    def select(self, choices: Iterable[Card]) -> Card:
        '''Select the node that maximizes the UCB score'''
        C = self.C(self.node.n)
        max_score = -sys.maxsize
        card: Card = None
        found = False
        for c in choices:
            for node in self.node.children:
                if str(node.card) == str(c):
                    found = True
                    if self.train:
                        val = node.score(C)
                    else:
                        val = node.avg_value()
                    if val > max_score:
                        max_score = val
                        card = node.card

        if not found:
            raise ValueError('None of choices represented in child nodes.')

        return card

    def advance(self, action: Card):
        '''Transitions to the next node, if it exists'''
        for child in self.node.children:
            if str(child.card) == str(action):
                self._node = child

                if self._node.n == 0:
                    self._in_tree = False

                return True
        self._in_tree = False
        return False
