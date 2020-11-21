import logging
import os
import time
from argparse import ArgumentParser
from typing import List

import numpy as np
import torch
from tqdm import tqdm

from ai import MCTS
from aiconfig import data_dir, model_dir
from aiutils import load, save
from buyagenda import BigMoneyBuyAgenda, TDBigMoneyBuyAgenda
from config import GameConfig
from enums import Rollout, StartingSplit
from game import Game
from mlp import SandboxMLP
from player import HeuristicPlayer, MCTSPlayer, MLPPlayer, RandomPlayer
from rollout import LinearRegressionRollout, RandomRollout
from simulationdata import SimulationData
from supply import Supply


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


def init_players(args: ArgumentParser, **kwargs):
    players = []
    j = 0
    config = kwargs['config']
    for i in range(args.players):
        if args.strategy[i] == 'R':
            players.append(RandomPlayer())
        elif args.strategy[i] == 'UCT':
            try:
                rollout_model = load(os.path.join(args.model_dir, args.rollouts[j]))
            except ImportError:
                rollout_model = RandomRollout()
            root = load(os.path.join(args.model_dir, args.roots[j]))
            j += 1
            uct_agent = MCTSPlayer(rollout_model, root=root)
            players.append(uct_agent)
        elif args.strategy[i] == 'BM':
            players.append(HeuristicPlayer(BigMoneyBuyAgenda()))
        elif args.strategy[i] == 'TDBM':
            players.append(HeuristicPlayer(TDBigMoneyBuyAgenda()))
        elif args.strategy[i] == 'MLP':
            model = SandboxMLP(config.feature_size, (config.feature_size + 1) // 2, 1)
            model.load_state_dict(torch.load(args.path))
            model.cuda()
            players.append(MLPPlayer(model, train=False))

    return players


def simulate(args: ArgumentParser, split: StartingSplit, n: int, save_data=False):

    sim_data = SimulationData()
    config = GameConfig(split, prosperity=args.prosperity, num_players=args.players, sandbox=args.sandbox)
    supply = Supply(config)
    players = init_players(args, config=config, supply=supply)

    for i in tqdm(range(n)):
        dominion = Game(config, players)
        dominion.new_game()

        for i, player in enumerate(players):
            if isinstance(player, MCTSPlayer):
                player.reset(dominion.state.player_states[i])

        t_start = time.time()
        dominion.run(T=args.T)
        t_end = time.time()
        sim_data.update(dominion, t_end - t_start)

    sim_data.finalize(dominion)

    if save_data:
        save(os.path.join(data_dir, args.data_name), sim_data)

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

    simulate(args, split, args.iters, args.save_data)


if __name__ == '__main__':
    parser = ArgumentParser('Simulation Chamber for Dominion')
    parser.add_argument('-T', type=int, default=None, help='Upper threshold for number of turns in each game')
    parser.add_argument('--iters', type=int, required=True, help='Number of games to simulate')
    parser.add_argument('--sandbox', action='store_true', help='When set, the supply is limited to the 7 basic kingdom supply cards.')
    parser.add_argument('--split', default=0, type=int, help='Starting Copper/Estate split. 0: Random, 1: 25Split, 2: 34Split')
    parser.add_argument('--prosperity', action='store_true', help='Whether the Prosperity settings should be used')
    parser.add_argument('--players', default=2, type=int, help='Number of AI players')
    parser.add_argument('--strategy', nargs='+', type=str, choices=['R', 'BM', 'TDBM', 'UCT', 'MLP'], help='Strategy of AI opponent.')
    parser.add_argument('--model_dir', default=model_dir, help='Where the models are located')
    parser.add_argument('--roots', nargs='+', help='Roots of UCT')
    parser.add_argument('--rollouts', nargs='+', help='Rollout models of UCT')
    parser.add_argument('--save_data', action='store_true', help='Whether the data should be saved')
    parser.add_argument('--data_dir', default=data_dir, type=str, help='Where the data should be saved')
    parser.add_argument('--data_name', default='data', type=str, help='Name of the data file')
    parser.add_argument('--debug', action='store_true', help='Turn logging settings to DEBUG')
    parser.add_argument('--path', default='../models/mlp_l1_default', help='Path to MLP model')

    args = parser.parse_args()
    main(args)
