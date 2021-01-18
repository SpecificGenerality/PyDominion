import logging
import random
import sys
from abc import ABC
from collections import Counter, defaultdict
from typing import List, Tuple

import numpy as np
import torch
from sklearn.linear_model import LinearRegression

from aiutils import load, save, softmax, update_mean
from card import Card
from enums import Zone
from mlp import MLP, BuyMLP
from state import State
from supply import Supply
from victorycard import Province


class RolloutModel(ABC):
    def update(self, **data):
        pass

    def select(self, choices: List[Card], **kwargs):
        pass

    def augment_data(self, data: dict):
        pass

    @classmethod
    def load(cls, **kwargs):
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

    @classmethod
    def load(cls, **kwargs):
        return cls()

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
        # D = softmax(scores)
        # choice = np.random.choice(choices, p=D)

        # Epsilon greedy
        if np.random.rand() < 0.05:
            choice = np.random.choice(choices)
        else:
            choice = choices[np.argmax(scores)]

        return choice

    def augment_data(self, data):
        return


class BuyMLPRollout(RolloutModel):
    def __init__(self, model: BuyMLP):
        self.model = model

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

        scores = self.model(s.feature.to_tensor())
        max_score = -sys.maxsize
        choice = None

        for card in choices:
            if card is None:
                idx = len(s.feature.idxs)
            else:
                idx = s.feature.idxs[str(card)]
            score = scores[idx]
            if score > max_score:
                max_score = score
                choice = card

        return choice


class HistoryHeuristicRollout(RolloutModel):
    def __init__(self, mast=defaultdict(dict), tau=0.5, train=False):
        self.mast = mast
        self.tau = tau
        self.train = train

    @classmethod
    def load(cls, **kwargs):
        path = kwargs['path']
        state_dict = load(path)
        mast = state_dict['mast']
        tau = state_dict['tau']
        return cls(mast, tau, False)

    def save(self, path: str):
        state_dict = {'mast': self.mast, 'tau': self.tau}
        save(path, state_dict)

    def update(self, **kwargs):
        '''Update history heuristic with card buys from last rollout'''
        data: List[Tuple[int, int, Card]] = kwargs['cards']
        score: int = kwargs['score']

        for agent_provinces, opp_provinces, coins, c in data:
            coins = max(min(coins, 8), 2)
            submast = self.mast[(agent_provinces, opp_provinces, coins)]
            if str(c) in submast:
                n = submast[str(c)][1]
                prev_mean = submast[str(c)][0]
                submast[str(c)] = (update_mean(n + 1, prev_mean, score), n + 1)
            else:
                submast[str(c)] = (score, 1)

    def select(self, choices, **kwargs):
        '''Create Gibbs distribution over choices given mast and return card choice'''
        state: State = kwargs['state']
        num_coins = state.player_states[0].get_total_coin_count(Zone.Play)
        agent_counts, opp_counts = state.get_player_card_counts(0), state.get_player_card_counts(1)
        # n_provinces = state.supply[Province]
        num_coins = max(min(8, num_coins), 2)
        submast = self.mast[(agent_counts[str(Province())], opp_counts[str(Province())], num_coins)]

        D = np.zeros(len(choices))
        for i, c in enumerate(choices):
            avg_reward, n = submast.get(str(c), (-1, 0))
            # Always choose any previously unexplored action
            if n == 0 and self.train:
                return c
            D[i] = avg_reward / self.tau

        if not self.train:
            return choices[np.argmax(D)]

        D = softmax(D)

        return np.random.choice(choices, p=D)

    # TODO: Deprecate/fix
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

    @classmethod
    def load(cls, **kwargs):
        raise NotImplementedError()

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


def load_rollout(rollout_type: str, model: str) -> RolloutModel:
    r_type_lower = rollout_type.lower()
    if r_type_lower == 'r':
        return RandomRollout()
    elif r_type_lower == 'hh':
        return HistoryHeuristicRollout.load(path=model)
    elif r_type_lower == 'mlp':
        return MLPRollout.load(path=model)

    raise ValueError(f'Invalid rollout type: {rollout_type}')


def init_rollouts(rollout_types: List[str], **kwargs) -> List[RolloutModel]:
    rollouts = []

    for r_type in rollout_types:
        r_type_lower = r_type.lower()
        if r_type_lower == 'r':
            rollouts.append(RandomRollout())
        elif r_type_lower == 'hh':
            rollouts.append(HistoryHeuristicRollout(train=True))
        elif r_type_lower == 'mlp':
            rollouts.append(MLPRollout(mlp=MLP(D_in=kwargs['D_in'], H=kwargs['H'])))

    return rollouts
