import pickle
from argparse import ArgumentParser
from typing import Iterable, Tuple

import numpy as np
from config import GameConfig
from enums import StartingSplit
from env import DefaultEnvironment
from player import Player, load_players
from sklearn.linear_model import LinearRegression
from state import DecisionResponse, State
from tqdm import tqdm


def train(n: int, sandbox: bool, players: Iterable[Player]) -> Tuple[LinearRegression, float]:
    config = GameConfig(split=StartingSplit.StartingRandomSplit, prosperity=False, num_players=len(players), sandbox=sandbox)
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

    reg = LinearRegression().fit(X, y)
    return reg, reg.score(X, y)


def predict(reg: LinearRegression, n: int, sandbox: bool, players: Iterable[Player]):
    config = GameConfig(split=StartingSplit.StartingRandomSplit, prosperity=False, num_players=len(players), sandbox=sandbox)
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

    return reg.score(X, y)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-n', default=10000, type=int, help='Number of games; number of rows in X')
    parser.add_argument('-path', type=str, help='Path to save (load) linreg model for train (predict)')
    parser.add_argument('--sandbox', action='store_true', help='Uses no action cards when set.')
    parser.add_argument('--predict', action='store_true', help='Predicts using linear regressor and random rollouts')
    parser.add_argument('--players', nargs='+', type=str, choices=['R', 'BM', 'TDBM', 'UCT', 'MLP'], help='AI strategy')
    parser.add_argument('--models', nargs='+', type=str, help='Paths to AI strategy models')
    args = parser.parse_args()

    players = load_players(args.players, args.models)

    if not args.predict:
        reg, score = train(args.n, args.sandbox, players)

        print(f'R_sq = {score}')

        pickle.dump(reg, open(args.path, 'wb'))
    else:
        reg: LinearRegression = pickle.load(open(args.path, 'rb'))
        score = predict(reg, args.n, args.sandbox, players)

        print(f'R_sq = {score}')
