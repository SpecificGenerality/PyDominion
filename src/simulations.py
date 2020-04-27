import json
import os
import time
from argparse import ArgumentParser
from typing import Dict, List

import numpy as np
from tqdm import tqdm

from ai import *
from aiutils import *
from buyagenda import *
from config import GameConfig
from enums import StartingSplit
from game import Game
from gamedata import GameData
from player import *
from simulationdata import *
from victorycard import *


def test_tau(taus: List, trials=100, iters=500):
    '''Test the UCT for varying values of tau'''
    agent = MCTS(30)
    for tau in taus:
        for _ in range(trials):
            agent.player=MCTSPlayer(train=True, tau=tau)
            agent.train(iters, trials)
            agent.data.update_dataframes()
            agent.data.augment_avg_scores(100)
    save(os.path.join(data_dir, 'taus'), agent.data)

def init_players(args: ArgumentParser):
    players = []
    j = 0
    for i in range(args.players):
        if args.strategy[i] == 'R':
            players.append(RandomPlayer())
        elif args.strategy[i] == 'UCT':
            try:
                rollout_model = load(os.path.join(args.model_dir, args.rollout[j]))
            except:
                rollout_model = RandomRollout()
            root = load(os.path.join(args.model_dir, args.roots[j]))
            j += 1
            uct_agent = MCTSPlayer(rollout_model, root=root)
            players.append(uct_agent)
        elif args.strategy[i] == 'BM':
            players.append(HeuristicPlayer(BigMoneyBuyAgenda()))
        elif args.strategy[i] == 'TDBM':
            players.append(HeuristicPlayer(TDBigMoneyBuyAgenda()))

    return players

def simulate(args: ArgumentParser, config: GameConfig, n: int, save_data=False):
    sim_data = SimulationData()

    players = init_players(args)

    for i in tqdm(range(n)):
        data = GameData(config)
        dominion = Game(config, data, players)
        dominion.newGame()

        for i, player in enumerate(players):
            if isinstance(player, MCTSPlayer):
                player.reset(dominion.state.playerStates[i])

        t_start = time.time()
        dominion.run()
        t_end = time.time()
        sim_data.update(dominion, t_end - t_start)

    sim_data.finalize()

    if save_data:
        save(os.path.join(data_dir, args.data_name), sim_data)

def main(args: ArgumentParser):
    if args.split == 0:
        split = StartingSplit.StartingRandomSplit
    elif args.split == 1:
        split = StartingSplit.Starting25Split
    else:
        split = StartingSplit.Starting34Split

    config = GameConfig(split, prosperity=args.prosperity, numPlayers=args.players)

    simulate(args, config, args.iters, args.save_data)


if __name__=='__main__':
    parser = ArgumentParser('Simulation Chamber for Dominion')
    parser.add_argument('--iters', type=int,  required=True, help='Number of games to simulate')
    parser.add_argument('--split', default=0, type=int, help='Starting Copper/Estate split. 0: Random, 1: 25Split, 2: 34Split')
    parser.add_argument('--prosperity', action='store_true', help='Whether the Prosperity settings should be used')
    parser.add_argument('--players', default=2, type=int, help='Number of AI players')
    parser.add_argument('--strategy', nargs='+', type=str, help='Strategy of AI opponent. Supported: [R, BM, TDBM, UCT]')
    parser.add_argument('--model_dir', default=model_dir, help='Where the models are located')
    parser.add_argument('--roots', nargs='+', help='Roots of UCT')
    parser.add_argument('--rollout', nargs='+', help='Rollout models of UCT')
    parser.add_argument('--save_data', action='store_true', help='Whether the data should be saved')
    parser.add_argument('--data_dir', default=data_dir, type=str, help='Where the data should be saved')
    parser.add_argument('--data_name', default='data', type=str, help='Name of the data file')

    args = parser.parse_args()
    main(args)
