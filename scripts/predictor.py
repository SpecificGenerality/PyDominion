import pickle
from argparse import ArgumentParser
from typing import Iterable, Tuple

import numpy as np
import torch
import torch.nn as nn
from config import GameConfig
from enums import FeatureType, StartingSplit
from env import DefaultEnvironment
from mlp import MLP
from player import Player, load_players
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from state import DecisionResponse, State
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


def sample_training_batch(n: int, p: float, config: GameConfig, players: Iterable[Player], one_hot=False) -> Tuple[np.array, np.array]:
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

            if rng.uniform(0, 1) < p:
                X.append(obs.feature.to_numpy())

        X.append(obs.feature.to_numpy())

        # TODO: Fix this hardcode
        if one_hot:
            label = np.zeros(3)
            label[(reward + 1)] = 1
            y.append(label)
        else:
            y.extend([reward + 1] * (len(X) - len(y)))

    return np.array(X), np.array(y)


def train_mlp(X, y, model: nn.Module, epochs: int, **kwargs) -> float:
    save_epochs = kwargs['save_epochs']
    path = kwargs['path']
    dataset = []

    print('Generating dataset for dataloader...')
    for i in tqdm(range(len(X))):
        dataset.append((torch.tensor(X[i]).cuda(), torch.tensor(y[i]).cuda()))

    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters())

    model.cuda()
    model.train()
    writer = SummaryWriter()

    print('Training MLP...')

    for epoch in tqdm(range(epochs)):
        running_loss = 0.0
        for i, data in enumerate(dataloader):
            inputs, labels = data

            optim.zero_grad()
            y_pred = model.forward(inputs)
            loss = criterion(y_pred, labels)
            loss.backward()
            optim.step()

            running_loss += loss.item()

        if save_epochs > 0 and epoch % save_epochs == 0:
            torch.save(model, path)

        writer.add_scalar("Loss/train", running_loss, epoch)

    writer.flush()
    return test_mlp(X, y, model)


def test_mlp(X: np.array, y: np.array, model: nn.Module) -> float:
    model.eval()
    inputs = torch.tensor(X).cuda()
    labels = torch.tensor(y).cuda()

    with torch.no_grad():
        y_pred = model(inputs)

    return (torch.argmax(y_pred, dim=1) == labels).sum().item() / len(y)


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

    players = load_players(args.players, args.models, config)

    if not args.predict:
        if args.reg_cls == MLP:
            X, y = sample_training_batch(args.n, args.p, config, players)
            model = MLP(config.feature_size, (config.feature_size + 1) // 2)
            acc = train_mlp(X, y, model, args.epochs, save_epochs=args.save_epochs, path=args.path)
            print(f'Acc = {acc}')
            torch.save(model, args.path)
        else:
            X, y = sample_training_batch(args.n, args.p, config, players)
            reg = load_predict_model(args.reg_cls, alpha=args.alpha)
            score = train_linear_model(X, y, reg)
            print(f'R_sq = {score}')
            pickle.dump(reg, open(args.path, 'wb'))
    else:
        if args.reg_cls == MLP:
            model = torch.load(args.path)
        else:
            reg = pickle.load(open(args.path, 'rb'))

        X, y = sample_training_batch(args.n, args.p, config, players)

        if args.reg_cls == MLP:
            acc = test_mlp(X, y, model)
            print(f'Acc = {acc}')
        else:
            score = reg.score(X, y)
            print(f'R_sq = {score}')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-n', required=True, default=10000, type=int, help='Number of games; number of rows in X')
    parser.add_argument('-path', required=True, type=str, help='Path to save (load) model for train (predict)')
    parser.add_argument('-p', required=True, default=-1, type=float, help='Sampling probability for each turn. Pass negative value to only sample terminal state.')
    parser.add_argument('-ftype', required=True, type=lambda x: {'full': FeatureType.FullFeature, 'reduced': FeatureType.ReducedFeature}.get(x.lower()))
    parser.add_argument('-reg-cls', type=lambda x: {'ols': LinearRegression, 'ridge': Ridge, 'logistic': LogisticRegression, 'mlp': MLP}.get(x.lower()))
    parser.add_argument('--sandbox', action='store_true', help='Uses no action cards when set.')
    parser.add_argument('--predict', action='store_true', help='Predicts using linear regressor and random rollouts')
    parser.add_argument('--players', nargs='+', type=str, choices=['R', 'BM', 'TDBM', 'UCT', 'MLP'], help='AI strategy')
    parser.add_argument('--models', nargs='+', type=str, help='Paths to AI strategy models')
    parser.add_argument('--epochs', type=int, help='Number of training epochs.')
    parser.add_argument('--save-epochs', type=int, default=0, help='Number of epochs between saves')
    parser.add_argument('--alpha', default=None, type=float, help='Ridge regression parameter')

    args = parser.parse_args()

    main(args)
