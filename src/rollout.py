import logging
import random
from abc import ABC
from collections import Counter
from typing import List

import numpy as np
import torch
from sklearn.linear_model import LinearRegression

from aiutils import softmax
from card import Card
from mlp import MLP
from state import State
from supply import Supply


class RolloutModel(ABC):
    def update(self, **data):
        pass

    def select(self, choices: List[Card], **kwargs):
        pass

    def augment_data(self, data: dict):
        pass


class RandomRollout(RolloutModel):
    def update(self, **data):
        '''Nothing to update'''
        pass

    def select(self, choices: List[Card], **kwargs):
        '''Select a card from choices uniformly at random'''
        return random.choice(choices)

    def augment_data(self, data):
        pass

    def __str__(self):
        return 'RandomRollout'


class MLPRollout(RolloutModel):
    def __init__(self, mlp: MLP):
        self.model = mlp

    @classmethod
    def load(cls, **kwargs):
        path = kwargs['path']
        model = torch.load(path)
        model.cuda()
        model.eval()
        return cls(model)

    def update(self, **data):
        return

    def select(self, choices, **kwargs):
        s: State = kwargs['state']
        p: int = s.player

        # Calculate values of next-step states
        X = s.lookahead_batch_featurize(choices)
        label_idx = 0 if p == 1 else 2
        y_pred = self.model.forward(X)
        scores = y_pred[:, label_idx].detach().cpu().numpy()

        # Softmax over scores
        D = softmax(scores)

        return np.random.choice(choices, p=D)

    def augment_data(self, data):
        return


class HistoryHeuristicRollout(RolloutModel):
    def __init__(self, tau=0.5, train=False):
        self.mast = {}
        self.tau = tau
        self.train = train

    def update(self, **data):
        '''Update history heuristic with card buys from last rollout'''
        cards: List[Card] = data['cards']
        score: int = data['score']

        for c in cards:
            if str(c) in self.mast:
                n = self.mast[str(c)][1]
                x_bar = self.mast[str(c)][0]
                self.mast[str(c)] = (x_bar / (n + 1) * n + score / (n + 1), n + 1)
            else:
                self.mast[str(c)] = (score, 1)

    def select(self, choices, **kwargs):
        '''Create Gibbs distribution over choices given mast and return card choice'''
        D = [np.exp(self.mast.get(str(c), (50 if self.train else 0, 0))[0] / self.tau) for c in choices]
        D /= sum(D)

        return np.random.choice(choices, p=D)

    def augment_data(self, data):
        '''Add the tau parameter and mast values to dict'''
        data['tau'] = self.tau
        for k, v in self.mast.items():
            data[f'Q({k})'] = v[0]

    def __str__(self):
        return 'HistoryHeuristicRollout'


class LinearRegressionRollout(RolloutModel):
    def __init__(self, iters: int, G: Supply, tau=0.5, train=False, eps=10e-10):
        self.supply: List[str] = G.get_supply_card_types() + [str(None)]
        # Index map for kingdom card
        self.n = len(self.supply)
        self.indices = dict(zip(self.supply, [i for i in range(self.n)]))
        self.betas = np.zeros((self.n, ))
        self.X = np.zeros((iters, self.n))
        self.y = np.zeros((iters, ))
        self.tau = tau
        self.train = train
        self.eps = eps

    def update(self, **data):
        if not self.train:
            return
        '''Perform OLS of self.X against self.y if there's enough data'''
        # We expect counts to be a Counter
        counts: Counter = data['counts']
        score: int = data['score']
        i: int = data['i']

        # Featurize observation
        x = np.zeros(self.n, )

        for k, v in counts.items():
            x[self.indices[k]] = v

        # Update observations matrix
        self.X[i] = x

        # Update label
        self.y[i] = score

        if i < self.n:
            logging.warning("Not updating betas, obviously rank-deficient system.")
            return

        reg = LinearRegression().fit(self.X[:i + 1], self.y[:i + 1])

        beta_norm = np.linalg.norm(self.betas)
        if (not np.isclose(beta_norm, 0)) and abs(np.linalg.norm(reg.coef_) - beta_norm) / beta_norm < self.eps:
            self.train = False

        self.betas = reg.coef_

    def select(self, choices, **kwargs):
        '''Sample from Gibbs distribution over choices weighted by regression weights'''
        D = [np.exp(self.betas[self.indices[str(c)]] / self.tau) for c in choices]
        D /= sum(D)
        return np.random.choice(choices, p=D)

    def augment_data(self, data: dict):
        '''Add tau parameter and regression weights to dict'''
        data['tau'] = self.tau
        for i, b in enumerate(self.betas):
            data[f'b({self.supply[i]})'] = b

    def __str__(self):
        return 'LinearRegressionRollout'
