from argparse import ArgumentParser

import numpy as np
import torch.nn as nn
from aiutils import save
from buffer import Buffer
from buyagenda import BigMoneyBuyAgenda
from config import GameConfig
from enums import StartingSplit, Zone
from env import DefaultEnvironment, Environment
from mcts import GameTree
from mlp import MLP
from player import HeuristicPlayer, MCTSPlayer, Player
from state import DecisionResponse, DecisionState, FeatureType, State
from tqdm import tqdm
from victorycard import Province

from mlprunner import train_mlp


def train_mcts(env: Environment, tree: GameTree, epochs: int, train_epochs: int, train_epochs_interval: int, **kwargs):
    save_epochs = kwargs['save_epochs']
    path = kwargs.pop('path')
    rollout_path = kwargs.pop('rollout_path')

    buf = Buffer(capacity=kwargs.pop('capacity'))
    # +3: province counts and coins, +1: None
    D_in = env.game.config.num_cards + 4
    # w/l/t
    D_out = 3
    H = (D_in + D_out) // 2

    model = MLP(D_in, H)

    for epoch in tqdm(range(epochs)):
        state: State = env.reset()
        tree.reset(state)
        done = False
        expanded = False
        flip = False
        data = {'features': [], 'labels': []}
        while not done:
            action = DecisionResponse([])
            d: DecisionState = state.decision
            player: Player = env.players[d.controlling_player] if tree.in_tree else env.players[1]

            # Add any states now visible due to randomness
            if tree.in_tree:
                cards = d.card_choices + [None]
                tree.node.add_unique_children(cards)

            player.makeDecision(state, action)

            agent_counts, opp_counts = state.get_player_card_counts(0), state.get_player_card_counts(1)
            if d.controlling_player == 0:
                x = np.zeros(D_in)
                x[:3] = [agent_counts[str(Province())], opp_counts[str(Province())], state.player_states[0].get_total_coin_count(Zone.Play)]
                # one-hot
                offset = 3
                if action.single_card is not None:
                    x[offset + state.feature.idxs[str(action.single_card)]] = 1
                else:
                    x[-1] = 1
                data['features'].append(x)

            # if tree.in_tree and d.controlling_player == 0:
            #     x = state.feature.to_numpy()
            #     L = [(node.card, node.n) for node in tree.node.children]
            #     cards, vals = zip(*L)
            #     y = Buffer.to_distribution(cards, state.feature.idxs, softmax(vals))
            #     buf.store(x, np.array(y, dtype=np.float32))

            # Advance to the next node within the tree, implicitly adding a node the first time we exit tree
            if tree.in_tree:
                tree.advance(action.single_card)

            # First time we go out of tree, enter rollout phase
            if not expanded and not tree.in_tree:
                # Previous node is starting player action, so current node is opponent player action.
                flip = (state.player == 1)
                expanded = True

            obs, reward, done, _ = env.step(action)

        buf.batch_store(np.array(data['features'], dtype=np.float32), [reward + 1] * len(data['features']))
        # delta = (state.get_player_score(0) - state.get_player_score(1)) * (-1 if flip else 1)
        delta = reward * (-1 if flip else 1)
        # env.players[0].rollout.update(**data)
        tree.node.backpropagate(delta)

        if save_epochs > 0 and epoch % save_epochs == 0:
            save(path, tree._root)
            # env.players[0].rollout.save(rollout_path)
        if (epoch + 1) % train_epochs_interval == 0:
            X, y = zip(*buf.buf)
            train_mlp(X, y, model, nn.CrossEntropyLoss(), epochs=train_epochs, save_epochs=20, path=rollout_path)

    save(path, tree._root)


def main(args):
    config = GameConfig(split=StartingSplit.StartingRandomSplit, prosperity=False, num_players=2, sandbox=args.sandbox, feature_type=args.ftype, device=args.device)

    tree = GameTree(train=True)

    player = MCTSPlayer(rollout=None, tree=tree, C=lambda x: np.sqrt(args.C))
    opponent = HeuristicPlayer(agenda=BigMoneyBuyAgenda())
    players = [player, opponent]

    env = DefaultEnvironment(config, players)

    train_mcts(env, tree, args.n, save_epochs=args.save_epochs, train_epochs=args.train_epochs, train_epochs_interval=args.train_epochs_interval, path=args.path, rollout_path=args.rollout_path, capacity=args.buffer_cap)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-n', default=10000, type=int, help='Number of training iterations')
    parser.add_argument('-ftype', required=True, type=lambda x: {'full': FeatureType.FullFeature, 'reduced': FeatureType.ReducedFeature}.get(x.lower()))
    parser.add_argument('-rollout', nargs='+', type=str, help='Type of rollout')
    parser.add_argument('-rollout-path', type=str, help='Path to save rollout model')
    parser.add_argument('-path', type=str, help='Path to save MCTS model')
    parser.add_argument('-C', default=2, type=float, help='Exploration constant')
    parser.add_argument('--buffer-cap', default=10000, type=int, help='Capacity of replay buffer')
    parser.add_argument('--save-epochs', type=int, default=0, help='Number of epochs between saves')
    parser.add_argument('--train-epochs', type=int, default=30, help='Number of MLP training epochs')
    parser.add_argument('--train-epochs-interval', type=int, default=1000, help='How often to train MLP')
    parser.add_argument('--sandbox', action='store_true', help='Uses no action cards when set.')
    parser.add_argument('-device', default='cuda')

    args = parser.parse_args()
    main(args)
