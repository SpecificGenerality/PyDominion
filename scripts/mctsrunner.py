from argparse import ArgumentParser

import numpy as np
from aiutils import save
from config import GameConfig
from enums import StartingSplit
from env import DefaultEnvironment, Environment
from player import MCTSPlayer
from rollout import MLPRollout
from state import DecisionResponse, DecisionState, FeatureType, State
from tqdm import tqdm


def train_mcts(env: Environment, epochs: int, **kwargs):
    save_epochs = kwargs['save_epochs']
    path = kwargs['path']
    player: MCTSPlayer = env.players[0]

    for epoch in tqdm(range(epochs)):
        state: State = env.reset()
        done = False
        expanded = False
        flip = False
        while not done:
            action = DecisionResponse([])
            d: DecisionState = state.decision
            player.makeDecision(state, action)

            if not expanded:
                cards = d.card_choices + [None]
                player.node.add_unique_children(cards)
                player.node = player.node.get_child_node(action.single_card)
                # Previous node is player zero action, so current node is player one action.
                flip = (state.player == 1)
                expanded = True

            obs, reward, done, _ = env.step(action)

        delta = reward * (1 if flip else -1)
        player.node.backpropagate(delta)

        if save_epochs > 0 and epoch % save_epochs == 0:
            save(path, player.root)


def main(args):
    config = GameConfig(split=StartingSplit.StartingRandomSplit, prosperity=False, num_players=2, sandbox=args.sandbox, feature_type=args.ftype, device=args.device)

    rollout = MLPRollout.load(path=args.rollout)

    player = MCTSPlayer(rollout=rollout, train=True, C=lambda x: np.sqrt(2))

    players = [player, player]

    env = DefaultEnvironment(config, players)

    train_mcts(env, args.n, save_epochs=args.save_epochs, path=args.path)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-n', default=10000, type=int, help='Number of training iterations')
    parser.add_argument('-ftype', required=True, type=lambda x: {'full': FeatureType.FullFeature, 'reduced': FeatureType.ReducedFeature}.get(x.lower()))
    parser.add_argument('-rollout', type=str, help='Path to rollout model')
    parser.add_argument('-path', type=str, help='Path to save MCTS model')
    parser.add_argument('--save-epochs', type=int, default=0, help='Number of epochs between saves')
    parser.add_argument('--sandbox', action='store_true', help='Uses no action cards when set.')
    parser.add_argument('-device', default='cuda')

    args = parser.parse_args()
    main(args)
