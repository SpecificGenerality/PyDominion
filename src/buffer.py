from collections import deque
from typing import Iterable, List, Mapping

import numpy as np

from card import Card


class Buffer:
    @classmethod
    def to_distribution(cls, cards: Iterable[Card], idxs: Mapping[str, int], vals: List[float]) -> np.array:
        # TODO: Fix this hardcode None option
        x = np.zeros(len(idxs) + 1)
        for i, card in enumerate(cards):
            if card is None:
                x[len(idxs)] = vals[i]
            else:
                j = idxs[str(card)]
                x[j] = vals[i]

        return x

    def __init__(self, capacity=100000):
        self.buf = deque(maxlen=capacity)
        self.capacity = capacity

    def store(self, x: np.array, D: np.array):
        self.buf.append([x, D])

    def batch_store(self, X: List[np.array], D: List[np.array]):
        if len(X) != len(D):
            raise ValueError('Length mismatch between data and labels.')

        n = len(X)
        for i in range(n):
            self.store(X[i], D[i])
