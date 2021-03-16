import os
from argparse import ArgumentParser

from aiutils import save
from config import GameConfig
from env import DefaultEnvironment, Environment
from mcts import GameTree
from player import MCTSPlayer, Player
from rollout import init_rollouts
from state import DecisionResponse, DecisionState, FeatureType, State
from tqdm import tqdm


def train_mcts(env: Environment, tree: GameTree, path: str, rollout_path: str, epochs: int, train_epochs_interval: int = 1000, save_epochs=1000, scoring='win_loss'):
    for epoch in tqdm(range(epochs)):
        state: State = env.reset()
        tree.reset(state)
        done = False
        expanded = False
        flip = False
        data = {'features': [], 'rewards': [], 'cards': [], 'idxs': state.feature.idxs}
        data['model_name'] = os.path.split(path)[-1]
        while not done:
            action = DecisionResponse([])
            d: DecisionState = state.decision
            player: Player = env.players[d.controlling_player]

            # Add any states now visible due to randomness
            if tree.in_tree:
                cards = d.card_choices + [None]
                tree.node.add_unique_children(cards)

            player.makeDecision(state, action)

            if isinstance(player, MCTSPlayer):
                x = state.feature.to_numpy()
                data['features'].append(x)
                data['cards'].append(action.single_card)

            # Advance to the next node within the tree, implicitly adding a node the first time we exit tree
            if tree.in_tree:
                tree.advance(action.single_card)

            # First time we go out of tree, enter rollout phase
            if not expanded and not tree.in_tree:
                # Previous node is starting player action, so current node is opponent player action.
                flip = (state.player == 1)
                expanded = True

            obs, reward, done, _ = env.step(action)

        data['rewards'].extend([reward] * (len(data['features']) - len(data['rewards'])))
        start_idx = 1 if flip else 0
        p0_score, p1_score = state.get_player_score(0), state.get_player_score(1)
        if scoring == 'score':
            p0_reward, p1_reward = p0_score, p1_score
        elif scoring == 'win_loss':
            if state.is_winner(0) and state.is_winner(1):
                p0_reward, p1_reward = 1 / 2, 1 / 2
            elif state.is_winner(0):
                p0_reward, p1_reward = 1, 0
            else:
                p0_reward, p1_reward = 0, 1
        elif scoring == 'score_ratio':
            min_score = min(p0_score, p1_score)
            p0_score_nonneg, p1_score_nonneg = p0_score + min_score, p1_score + min_score
            if p0_score_nonneg == 0 and p1_score_nonneg == 0:
                p0_reward, p1_reward = 0, 0
            else:
                total_score = p0_score_nonneg + p1_score_nonneg
                p0_reward, p1_reward = p0_score / total_score, p1_score / total_score

        tree.node.backpropagate((p0_reward, p1_reward), start_idx=start_idx)

        if save_epochs > 0 and epoch % save_epochs == 0:
            save(path, tree._root)

            for player in env.players:
                if isinstance(player, MCTSPlayer):
                    player.rollout.save(rollout_path)
                    break

        # mcts players share the tree, so only update once
        for player in env.players:
            if isinstance(player, MCTSPlayer):
                player.rollout.update(**data)
                if (epoch + 1) % train_epochs_interval == 0:
                    player.rollout.learn()

    for player in env.players:
        if isinstance(player, MCTSPlayer):
            player.rollout.save(rollout_path)
            break
    save(path, tree._root)


def main(args):
    config = GameConfig(prosperity=False, num_players=2, sandbox=args.sandbox, feature_type=args.ftype, device=args.device)

    tree = GameTree(train=True)

    D_in = config.feature_size
    H = (config.feature_size + 1) // 2

    player = MCTSPlayer(rollout=init_rollouts(args.rollout, D_in=D_in, H=H)[0], tree=tree)

    players = [player, player]

    env = DefaultEnvironment(config, players)

    train_mcts(env, tree, args.n, save_epochs=args.save_epochs, train_epochs=args.train_epochs, train_epochs_interval=args.train_epochs_interval, path=args.path, rollout_path=args.rollout_path, capacity=args.buffer_cap)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-n', default=10000, type=int, help='Number of training iterations')
    parser.add_argument('-ftype', required=True, type=lambda x: {'full': FeatureType.FullFeature, 'reduced': FeatureType.ReducedFeature}.get(x.lower()))
    parser.add_argument('-opponent', type=str, choices=['H', 'LOG', 'R', 'BM', 'TDBM', 'UCT', 'MLP', 'GMLP'], help='Strategy of AI opponent.')
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
