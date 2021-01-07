from argparse import ArgumentParser

import numpy as np
from aiutils import save
from buyagenda import BigMoneyBuyAgenda
from config import GameConfig
from enums import StartingSplit
from env import DefaultEnvironment, Environment
from mcts import GameTree
from player import HeuristicPlayer, MCTSPlayer, Player
from rollout import MLPRollout, RandomRollout
from state import DecisionResponse, DecisionState, FeatureType, State
from tqdm import tqdm


def train_mcts(env: Environment, tree: GameTree, epochs: int, **kwargs):
    save_epochs = kwargs['save_epochs']
    path = kwargs['path']

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

    save(path, tree._root)


def main(args):
    config = GameConfig(split=StartingSplit.StartingRandomSplit, prosperity=False, num_players=2, sandbox=args.sandbox, feature_type=args.ftype, device=args.device)

    rollout = MLPRollout.load(path=args.rollout) if args.rollout else RandomRollout()

    tree = GameTree(train=True)

    player = MCTSPlayer(rollout=rollout, tree=tree, C=lambda x: np.sqrt(args.C))
    opponent = HeuristicPlayer(agenda=BigMoneyBuyAgenda())
    players = [player, opponent]

    env = DefaultEnvironment(config, players)

    train_mcts(env, tree, args.n, save_epochs=args.save_epochs, path=args.path)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-n', default=10000, type=int, help='Number of training iterations')
    parser.add_argument('-ftype', required=True, type=lambda x: {'full': FeatureType.FullFeature, 'reduced': FeatureType.ReducedFeature}.get(x.lower()))
    parser.add_argument('-rollout', type=str, help='Path to rollout model')
    parser.add_argument('-path', type=str, help='Path to save MCTS model')
    parser.add_argument('-C', default=2, type=float, help='Exploration constant')
    parser.add_argument('--save-epochs', type=int, default=0, help='Number of epochs between saves')
    parser.add_argument('--sandbox', action='store_true', help='Uses no action cards when set.')
    parser.add_argument('-device', default='cuda')

    args = parser.parse_args()
    main(args)
