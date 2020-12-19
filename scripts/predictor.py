import pickle
from argparse import ArgumentParser
from typing import Tuple

import numpy as np
from config import GameConfig
from enums import StartingSplit
from env import DefaultEnvironment
from player import RandomPlayer
from sklearn.linear_model import LinearRegression
from state import DecisionResponse, State
from tqdm import tqdm


def train(n: int, sandbox: bool) -> Tuple[LinearRegression, float]:
    config = GameConfig(split=StartingSplit.StartingRandomSplit, prosperity=False, num_players=2, sandbox=sandbox)
    players = [RandomPlayer(), RandomPlayer()]
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


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-n', default=10000, type=int, help='Number of games; number of rows in X')
    parser.add_argument('-path', type=str, help='Path to save linreg model')
    parser.add_argument('--sandbox', action='store_true', help='Uses no action cards when set.')

    args = parser.parse_args()
    reg, score = train(args.n, args.sandbox)

    print(f'R_sq = {score}')

    pickle.dump(reg, open(args.path, 'wb'))
