from argparse import ArgumentParser

import numpy as np
import torch.nn as nn
from aiutils import save, softmax
from buffer import Buffer
from buyagenda import BigMoneyBuyAgenda
from config import GameConfig
from enums import StartingSplit
from env import DefaultEnvironment, Environment
from mcts import GameTree
from mlp import BuyMLP
from player import HeuristicPlayer, MCTSPlayer, Player
from rollout import MLPRollout, RandomRollout
from state import DecisionResponse, DecisionState, FeatureType, State
from tqdm import tqdm

from mlprunner import train_mlp


def train_mcts(env: Environment, tree: GameTree, epochs: int, train_epochs: int, train_epochs_interval: int, **kwargs):
    save_epochs = kwargs['save_epochs']
    path = kwargs['path']
    mlp_path = kwargs['mlp_path']

    buf = Buffer(capacity=kwargs['capacity'])
    D_in = env.game.config.feature_size
    # +1 for None
    D_out = env.game.config.num_cards + 1
    H = (D_in + D_out) // 2

    model = BuyMLP(D_in, H, D_out)

    for epoch in tqdm(range(epochs)):
        state: State = env.reset()
        tree.reset(state)
        done = False
        expanded = False
        flip = False
        while not done:
            action = DecisionResponse([])
            d: DecisionState = state.decision
            player: Player = env.players[d.controlling_player] if tree.in_tree else env.players[1]

            # Add any states now visible due to randomness
            if tree.in_tree:
                cards = d.card_choices + [None]
                tree.node.add_unique_children(cards)

            player.makeDecision(state, action)

            if tree.in_tree and d.controlling_player == 0:
                x = state.feature.to_numpy()
                L = [(node.card, node.n) for node in tree.node.children]
                cards, vals = zip(*L)
                y = Buffer.to_distribution(cards, state.feature.idxs, softmax(vals))
                buf.store(x, np.array(y, dtype=np.float32))

            # Advance to the next node within the tree, implicitly adding a node the first time we exit tree
            if tree.in_tree:
                tree.advance(action.single_card)

            # First time we go out of tree, enter rollout phase
            if not expanded and not tree.in_tree:
                # Previous node is starting player action, so current node is opponent player action.
                flip = (state.player == 1)
                expanded = True

            obs, reward, done, _ = env.step(action)

        delta = (state.get_player_score(0) - state.get_player_score(1)) * (-1 if flip else 1)
        tree.node.backpropagate(delta)

        if save_epochs > 0 and epoch % save_epochs == 0:
            save(path, tree._root)

        if (epoch + 1) % train_epochs_interval == 0:
            X, y = zip(*buf.buf)
            train_mlp(X, y, model, nn.MSELoss(), epochs=train_epochs, save_epochs=20, path=mlp_path, lr=kwargs['alpha'])

    save(path, tree._root)


def main(args):
    config = GameConfig(split=StartingSplit.StartingRandomSplit, prosperity=False, num_players=2, sandbox=args.sandbox, feature_type=args.ftype, device=args.device)

    rollout = MLPRollout.load(path=args.rollout) if args.rollout else RandomRollout()

    tree = GameTree(train=True)

    player = MCTSPlayer(rollout=rollout, tree=tree, C=lambda x: np.sqrt(args.C))
    opponent = HeuristicPlayer(agenda=BigMoneyBuyAgenda())
    players = [player, opponent]

    env = DefaultEnvironment(config, players)

    train_mcts(env, tree, args.n, save_epochs=args.save_epochs, train_epochs=args.train_epochs, train_epochs_interval=args.train_epochs_interval, path=args.path, mlp_path=args.mlp_path, alpha=args.alpha, capacity=args.buffer_cap)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-n', default=10000, type=int, help='Number of training iterations')
    parser.add_argument('-ftype', required=True, type=lambda x: {'full': FeatureType.FullFeature, 'reduced': FeatureType.ReducedFeature}.get(x.lower()))
    parser.add_argument('-rollout', type=str, help='Path to rollout model')
    parser.add_argument('-path', type=str, help='Path to save MCTS model')
    parser.add_argument('-mlp-path', type=str, help='Path to save MLP policy.')
    parser.add_argument('-C', default=2, type=float, help='Exploration constant')
    parser.add_argument('--buffer-cap', default=10000, type=int, help='Capacity of replay buffer')
    parser.add_argument('--alpha', default=0.001, type=float, help="Learning rates")
    parser.add_argument('--save-epochs', type=int, default=0, help='Number of epochs between saves')
    parser.add_argument('--train-epochs', type=int, default=30, help='Number of MLP training epochs')
    parser.add_argument('--train-epochs-interval', type=int, default=1000, help='How often to train MLP')
    parser.add_argument('--sandbox', action='store_true', help='Uses no action cards when set.')
    parser.add_argument('-device', default='cuda')

    args = parser.parse_args()
    main(args)
