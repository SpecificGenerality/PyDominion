from argparse import ArgumentParser

import numpy as np
from aiutils import save
from config import GameConfig
from enums import StartingSplit
from env import DefaultEnvironment, Environment
from mcts import GameTree, Node
from player import MCTSPlayer
from rollout import MLPRollout
from state import DecisionResponse, DecisionState, FeatureType, State
from tqdm import tqdm


def train_mcts(env: Environment, tree: GameTree, epochs: int, **kwargs):
    save_epochs = kwargs['save_epochs']
    path = kwargs['path']
    player: MCTSPlayer = env.players[0]

    for epoch in tqdm(range(epochs)):
        state: State = env.reset()
        tree.reset(state)
        done = False
        expanded = False
        flip = False
        while not done:
            action = DecisionResponse([])
            d: DecisionState = state.decision
            next_node: Node = player.makeDecision(state, action)

            if not expanded and not next_node:
                cards = d.card_choices + [None]
                tree.node.add_unique_children(cards)
                # Previous node is starting player action, so current node is opponent player action.
                flip = (state.player == 1)
                expanded = True

            obs, reward, done, _ = env.step(action)
            tree.advance(action.single_card)

        delta = reward * (1 if flip else -1)
        tree.node.backpropagate(delta)

        if save_epochs > 0 and epoch % save_epochs == 0:
            save(path, tree._root)

    save(path, tree._root)


def main(args):
    config = GameConfig(split=StartingSplit.StartingRandomSplit, prosperity=False, num_players=2, sandbox=args.sandbox, feature_type=args.ftype, device=args.device)

    rollout = MLPRollout.load(path=args.rollout)

    tree = GameTree(train=True)

    player = MCTSPlayer(rollout=rollout, tree=tree, C=lambda x: np.sqrt(args.C))

    players = [player, player]

    env = DefaultEnvironment(config, players)

    train_mcts(env, tree, args.n, save_epochs=args.save_epochs, path=args.path)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-n', default=10000, type=int, help='Number of training iterations')
    parser.add_argument('-ftype', required=True, type=lambda x: {'full': FeatureType.FullFeature, 'reduced': FeatureType.ReducedFeature}.get(x.lower()))
    parser.add_argument('-rollout', type=str, help='Path to rollout model')
    parser.add_argument('-path', type=str, help='Path to save MCTS model')
    parser.add_argument('-C', type=float, help='Exploration constant')
    parser.add_argument('--save-epochs', type=int, default=0, help='Number of epochs between saves')
    parser.add_argument('--sandbox', action='store_true', help='Uses no action cards when set.')
    parser.add_argument('-device', default='cuda')

    args = parser.parse_args()
    main(args)
