import pickle
import sys
from argparse import ArgumentParser
from typing import Iterable

import numpy as np
from config import GameConfig
from enums import FeatureType, StartingSplit
from env import DefaultEnvironment
from player import Player, load_players
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from state import DecisionResponse, State
from tqdm import tqdm


def train(n: int, predict_model, config: GameConfig, players: Iterable[Player], **kwargs) -> float:
    env = DefaultEnvironment(config, players)

    X = np.zeros((n, len(env.game.state.feature)))
    y = np.zeros(n)

    for epoch in tqdm(range(n)):
        state: State = env.reset()
        done = False
        while not done:
            action = DecisionResponse([])
            d = state.decision
            player = players[d.controlling_player]
            player.makeDecision(state, action)
            obs, reward, done, _ = env.step(action)

        X[epoch] = obs.feature.to_numpy()
        y[epoch] = reward

    reg = predict_model.fit(X, y)
    return reg.score(X, y)


def predict(reg, n: int, config: GameConfig, players: Iterable[Player], turn_break=sys.maxsize):
    env = DefaultEnvironment(config, players)

    X = np.zeros((n, len(env.game.state.feature)))
    y = np.zeros(n)

    for epoch in tqdm(range(n)):
        state: State = env.reset()
        done = False
        feature_done = False
        while not done:
            action = DecisionResponse([])
            d = state.decision
            player = players[d.controlling_player]
            player.makeDecision(state, action)
            obs, reward, done, _ = env.step(action)

            if state.player_states[d.controlling_player].turns > turn_break:
                X[epoch] = obs.feature.to_numpy()
                feature_done = True

        if not feature_done:
            X[epoch] = obs.feature.to_numpy()
        y[epoch] = reward

    return reg.score(X, y)


def evaluate_turn_break(reg, n: int, config: GameConfig, players: Iterable[Player]) -> np.array:
    scores = []
    for i in range(2, 30, 2):
        score = predict(reg, n, config, players, turn_break=i)
        scores.append(score)
    return np.array(scores)


def load_predict_model(reg_cls, alpha: float = None):
    if reg_cls == LinearRegression:
        return reg_cls()
    elif reg_cls == Ridge:
        return reg_cls(alpha=alpha)
    elif reg_cls == LogisticRegression:
        if alpha:
            return reg_cls(max_iter=10e5, C=alpha)
        return reg_cls(max_iter=10e5)


def main(args: ArgumentParser):
    reg = load_predict_model(args.reg_cls, alpha=args.alpha)

    config = GameConfig(split=StartingSplit.StartingRandomSplit, prosperity=False, num_players=len(args.players), sandbox=args.sandbox, feature_type=args.ftype)

    players = load_players(args.players, args.models, config)

    if not args.predict:
        score = train(args.n, reg, config, players)

        print(f'R_sq = {score}')

        pickle.dump(reg, open(args.path, 'wb'))
    else:
        reg = pickle.load(open(args.path, 'rb'))
        # score = predict(reg, args.n, config, players)
        scores = evaluate_turn_break(reg, args.n, config, players)

        print(f'R_sq = {scores}')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-n', required=True, default=10000, type=int, help='Number of games; number of rows in X')
    parser.add_argument('-path', required=True, type=str, help='Path to save (load) linreg model for train (predict)')
    parser.add_argument('-ftype', required=True, type=lambda x: {'full': FeatureType.FullFeature, 'reduced': FeatureType.ReducedFeature}.get(x.lower()))
    parser.add_argument('-reg_cls', type=lambda x: {'ols': LinearRegression, 'ridge': Ridge, 'logistic': LogisticRegression}.get(x.lower()))
    parser.add_argument('--sandbox', action='store_true', help='Uses no action cards when set.')
    parser.add_argument('--predict', action='store_true', help='Predicts using linear regressor and random rollouts')
    parser.add_argument('--players', nargs='+', type=str, choices=['R', 'BM', 'TDBM', 'UCT', 'MLP'], help='AI strategy')
    parser.add_argument('--models', nargs='+', type=str, help='Paths to AI strategy models')
    parser.add_argument('--alpha', default=None, type=float, help='Ridge regression parameter')

    args = parser.parse_args()

    main(args)
