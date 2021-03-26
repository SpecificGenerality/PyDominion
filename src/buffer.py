from collections import deque
from typing import List, Tuple

import numpy as np


class Buffer:
    def __init__(self, capacity=20000, locked_capacity=0):
        self.buf = deque(maxlen=capacity)
        self.locked_buf = deque(maxlen=locked_capacity)
        self.capacity = capacity
        self.i = locked_capacity

    def store(self, x: np.array, D: np.array):
        self.buf.append([x, D])

    def locked_store(self, x: np.array, D: np.array):
        if len(self.locked_buf) == self.locked_buf.maxlen:
            return
        self.locked_buf.append([x, D])

    def batch_store(self, X: List[np.array], D: List[np.array]):
        if len(X) != len(D):
            raise ValueError('Length mismatch between data and labels.')

        n = len(X)
        for i in range(n):
            self.store(X[i], D[i])

    def unzip(self) -> Tuple[List[np.array], List[np.array]]:
        buf = self.buf + self.locked_buf
        return zip(*buf)

    def __len__(self):
        return len(self.buf) + len(self.locked_buf)
