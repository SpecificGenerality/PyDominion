import os
import pickle
from argparse import ArgumentParser
from typing import Iterable, Tuple

import numpy as np
import torch
import torch.nn as nn
from config import GameConfig
from enums import FeatureType, StartingSplit
from env import DefaultEnvironment
from mlp import PredictorMLP
from player import Player, load_players
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import confusion_matrix
from state import DecisionResponse, State
from tqdm import tqdm

from mlprunner import train_mlp


def sample_training_batch(n: int, p: float, config: GameConfig, players: Iterable[Player], win_loss=False) -> Tuple[np.array, np.array]:
    env = DefaultEnvironment(config, players)
    X = []
    y = []

    rng = np.random.default_rng()

    print('Generating training data from self-play...')
    for epoch in tqdm(range(n)):
        state: State = env.reset()
        done = False
        while not done:
            action = DecisionResponse([])
            d = state.decision
            player = players[d.controlling_player]
            player.makeDecision(state, action)
            obs, reward, done, _ = env.step(action)

            feature = obs.feature.to_numpy()
            if p <= 1 and p > 0:
                if rng.uniform(0, 1) < p:
                    X.append(feature)
            else:
                if obs.player_states[d.controlling_player].turns < p:
                    X.append(feature)

        if p <= 0:
            X.append(feature)

        y.extend([reward] * (len(X) - len(y)))

    y = np.array(y)

    if win_loss:
        y[y == -1] = 0

    return np.array(X), y


def test_mlp(X: np.array, y: np.array, model: nn.Module) -> float:
    model.eval()
    inputs = torch.tensor(X).cuda()
    labels = torch.tensor(y).cuda()

    with torch.no_grad():
        y_pred = model(inputs)

    y_pred_classes = torch.argmax(y_pred, dim=1)
    y_pred_classes_np = y_pred_classes.cpu().numpy()
    cm = confusion_matrix(y, y_pred_classes_np)

    return (y_pred_classes == labels).sum().item() / len(y), cm.diagonal() / cm.sum(axis=1)


def train_linear_model(X, y, predict_model) -> float:
    reg = predict_model.fit(X, y)

    return reg.score(X, y)


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
    config = GameConfig(split=StartingSplit.StartingRandomSplit, prosperity=False, num_players=len(args.players), sandbox=args.sandbox, feature_type=args.ftype)

    players = load_players(args.players, args.models, config, train=True)

    if not args.predict:
        X, y = sample_training_batch(args.n, args.p, config, players)

        if args.reg_cls == PredictorMLP:
            model = PredictorMLP(config.feature_size, (config.feature_size + 1) // 2, D_out=2)
            train_mlp(X, y, model, nn.CrossEntropyLoss(), args.epochs, save_epochs=args.save_epochs, path=args.path)
            acc = test_mlp(X, y, model)
            print(f'Acc = {acc}')
            torch.save(model, args.path)
        else:
            reg = load_predict_model(args.reg_cls, alpha=args.alpha)
            score = train_linear_model(X, y, reg)
            print(f'R_sq = {score}')
            pickle.dump(reg, open(args.path, 'wb'))

        np.savez(os.path.join('data', os.path.split(args.path)[-1]))
    else:
        if args.reg_cls == PredictorMLP:
            model = torch.load(args.path)
        else:
            reg = pickle.load(open(args.path, 'rb'))

        X, y = sample_training_batch(args.n, args.p, config, players)

        if args.reg_cls == PredictorMLP:
            acc, cl_acc = test_mlp(X, y, model)
            print(f'Acc = {acc}, {cl_acc}')
        else:
            score = reg.score(X, y)
            print(f'R_sq = {score}')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-n', required=True, default=10000, type=int, help='Number of games; number of rows in X')
    parser.add_argument('-path', required=True, type=str, help='Path to save (load) model for train (predict)')
    parser.add_argument('-p', required=True, default=-1, type=float, help='Sampling probability for each turn. Pass negative value to only sample terminal state.')
    parser.add_argument('-ftype', required=True, type=lambda x: {'full': FeatureType.FullFeature, 'reduced': FeatureType.ReducedFeature}.get(x.lower()))
    parser.add_argument('-reg-cls', type=lambda x: {'ols': LinearRegression, 'ridge': Ridge, 'logistic': LogisticRegression, 'mlp': PredictorMLP}.get(x.lower()))
    parser.add_argument('--sandbox', action='store_true', help='Uses no action cards when set.')
    parser.add_argument('--predict', action='store_true', help='Predicts using linear regressor and random rollouts')
    parser.add_argument('--players', nargs='+', type=str, choices=['R', 'BM', 'TDBM', 'UCT', 'MLP'], help='AI strategy')
    parser.add_argument('--models', nargs='+', type=str, help='Paths to AI strategy models')
    parser.add_argument('--epochs', type=int, help='Number of training epochs.')
    parser.add_argument('--save-epochs', type=int, default=0, help='Number of epochs between saves')
    parser.add_argument('--alpha', default=None, type=float, help='Ridge regression parameter')

    args = parser.parse_args()

    main(args)
