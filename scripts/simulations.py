import logging
import os
import time
from argparse import ArgumentParser
from typing import Iterable, List

import numpy as np
from ai import MCTS
from aiconfig import data_dir
from aiutils import save
from config import GameConfig
from enums import Rollout, StartingSplit
from game import Game
from mcts import GameTree
from player import MCTSPlayer, Player, load_players
from rollout import LinearRegressionRollout
from simulationdata import SimulationData
from state import FeatureType
from tqdm import tqdm


def test_tau(taus: List, trials=100, iters=500):
    '''Test the UCT for varying values of tau'''
    agent = MCTS(30, n=iters, tau=0.5, rollout=Rollout.LinearRegression, eps=0)
    for tau in taus:
        for _ in range(trials):
            agent.rollout = LinearRegressionRollout(iters, agent.supply, tau, train=True, eps=0)
            agent.player = MCTSPlayer(rollout=agent.rollout, train=True)
            agent.train(n=iters, output_iters=iters)
            agent.data.update_dataframes()
            agent.data.augment_avg_scores(100)
    save(os.path.join(data_dir, 'taus-lr'), agent.data)


def test_C(trials=10, iters=500):
    agent = MCTS(T=30, n=iters, tau=0.5, rollout=Rollout.LinearRegression, eps=0)
    L = [lambda x: 25, lambda x: 25 / np.sqrt(x), lambda x: max(1, min(25, 25 / np.sqrt(x)))]
    for C in L:
        for _ in range(trials):
            agent.rollout = LinearRegressionRollout(iters, agent.supply, train=True, eps=0)
            agent.player = MCTSPlayer(rollout=agent.rollout, train=True, C=C)
            agent.train(n=iters, output_iters=100)
            agent.data.update_dataframes()
            agent.data.augment_avg_scores(100)
    save(os.path.join(data_dir, 'C-lr'), agent.data)


def simulate(n: int, config: GameConfig, players: Iterable[Player], save_data=False, data_path=None):

    sim_data = SimulationData()

    for i in tqdm(range(n)):
        dominion = Game(config, players)
        dominion.new_game()

        for i, player in enumerate(players):
            if isinstance(player, MCTSPlayer):
                player.reset(p_state=dominion.state.player_states[i])

        t_start = time.time()
        dominion.run(T=args.T)
        t_end = time.time()
        sim_data.update(dominion, t_end - t_start)

    sim_data.finalize(dominion)

    if save_data:
        save(data_path, sim_data)

    print('===SUMMARY===')
    print(sim_data.summary)


def main(args: ArgumentParser):

    if args.debug:
        logging.basicConfig(level=logging.INFO)
    if args.split == 0:
        split = StartingSplit.StartingRandomSplit
    elif args.split == 1:
        split = StartingSplit.Starting25Split
    else:
        split = StartingSplit.Starting34Split

    config = GameConfig(split=split, prosperity=args.prosperity, num_players=len(args.players), sandbox=args.sandbox, feature_type=args.ftype, device=args.device)

    if args.tree_path:
        tree = GameTree.load(args.tree_path, False)
    else:
        tree = None

    players = load_players(args.players, args.models, config, tree=tree)
    simulate(args.n, config, players, args.save_data, args.data_path)


if __name__ == '__main__':
    parser = ArgumentParser('Simulation Chamber for Dominion')
    parser.add_argument('-n', type=int, required=True, help='Number of games to simulate')
    parser.add_argument('-T', type=int, default=None, help='Upper threshold for number of turns in each game')
    parser.add_argument('-ftype', required=True, type=lambda x: {'full': FeatureType.FullFeature, 'reduced': FeatureType.ReducedFeature}.get(x.lower()))
    parser.add_argument('--sandbox', action='store_true', help='When set, the supply is limited to the 7 basic kingdom supply cards.')
    parser.add_argument('--prosperity', action='store_true', help='Whether the Prosperity settings should be used')
    parser.add_argument('--split', default=0, type=int, help='Starting Copper/Estate split. 0: Random, 1: 25Split, 2: 34Split')
    parser.add_argument('--tree-path', type=str, help='Path to game tree.')
    parser.add_argument('--players', nargs='+', type=str, choices=['H', 'LOG', 'R', 'BM', 'TDBM', 'UCT', 'MLP', 'GMLP'], help='Strategy of AI opponent.')
    parser.add_argument('--device', default='cuda', type=str, help='Hardware to use for neural network models.')
    parser.add_argument('--models', nargs='+', type=str, help='Path to AI models')
    parser.add_argument('--save_data', action='store_true', help='Whether the data should be saved')
    parser.add_argument('--data_path', type=str, help='Where to save data file')
    parser.add_argument('--debug', action='store_true', help='Turn logging settings to DEBUG')

    args = parser.parse_args()

    main(args)
