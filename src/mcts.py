from collections import Counter
from typing import Dict, List
from state import *
import numpy as np
import sys

class MCTSState:
    def __init__(self, s: State):
        self.player = s.decision.controllingPlayer
        # In sandbox mode, hand is not important for making decisions
        # self.hand = Counter([str(c) for c in s.playerStates[self.player].hand])
        self.deck = Counter([str(c) for c in s.playerStates[self.player].getAllCards()])
        self.supply = Counter(s.data.supply)

    # Two MCTS states are equal if
    #   1) The hands are equal up to multiplicity
    #   2) The decks and supplies are equal after binning
    def __eq__(self, other):
        assert self.player == other.player, 'MCTSError: Different players in game tree'

        # if self.hand != other.hand:
        #     return False
        return self.deck == other.deck and self.supply == other.supply

class Node:
    def __init__(self, parent=None, card=None):
        # TODO: For non-sandbox, maybe states should be stored?
        # self.state = s
        # the card played to get from parent to current
        self.card = card
        # number of times visited
        self.n = 0
        # node value
        self.v = 0
        self.parent = parent
        self.children = []

    # UCB1 formula
    def score(self, C):
        return self.v + C * np.sqrt(np.log(self.parent.n) / self.n) if self.n > 0 else sys.maxsize

    def update_v(self, f):
        vals = [n.v for n in self.children if n.n > 0]
        self.v = f(vals)

    def add_unique_children(self, cards: List[Card]):
        for c in cards:
            found = False
            if isinstance(c, Curse):
                continue
            for child in self.children:
                if str(c) == str(child.card):
                    found = True
            if not found:
                self.children.append(Node(self, c))

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
        return f'{self.parent.card}<--n: {self.n}, v: {self.v}, c: {self.card}-->{[str(c.card) for c in self.children]}'

    def __repr__(self):
        return str(self)