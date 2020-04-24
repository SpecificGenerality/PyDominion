import json
import os
import time
from argparse import ArgumentParser
from typing import List

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

def simulate(args: ArgumentParser, config: GameConfig, n: int, save=False):
    print('Starting game simulations...')
    start_time = time.time()
    sim_stats = {'Iters': 0, 'StartWins': 0, 'Ties': 0, 'ProvinceWins': 0, 'Degenerate': 0}
    scores = np.zeros((n, config.numPlayers))
    mcts_player = MCTSPlayer(mast=load(os.path.join(data_dir, 'uct-default')).get_last_mast())

    for i in tqdm(range(n)):

        data = GameData(config)

        if args.strategy == 'Random':
            playerClass = RandomPlayer
        elif args.strategy == 'BigMoney':
            playerClass = HeuristicPlayer

        players = [mcts_player]

        dominion = Game(config, data, players)

        dominion.newGame()

        mcts_player.reset(dominion.state.playerStates[0])
        dominion.run(T=30)
        game_stats = dominion.getStats()
        sim_stats['Iters'] += 1
        sim_stats['StartWins'] += (1 if len(game_stats['Winners']) == 1 and game_stats['Winners'][0] == 0 else 0)
        sim_stats['Ties'] += (1 if len(game_stats['Winners']) > 1 else 0)
        sim_stats['ProvinceWins'] += 1 if any(isinstance(card, Province) for card in game_stats['EmptyPiles']) else 0
        sim_stats['Degenerate'] += 1 if dominion.state.isDegenerate() else 0
        scores[i] = dominion.getPlayerScores()

    if save:
        with open(os.path.join(data_dir, 'uct-default-1000-copper.txt'), 'w+') as file:
            json.dump(sim_stats, file)

        np.savez(os.path.join(data_dir, 'uct-default-1000-copper'), scores)


def main(args: ArgumentParser):
    if args.split == 0:
        split = StartingSplit.StartingRandomSplit
    elif args.split == 1:
        split = StartingSplit.Starting25Split
    else:
        split = StartingSplit.Starting34Split

    config = GameConfig(split, prosperity=args.prosperity, numPlayers=args.players)

    simulate(args, config, args.iters, args.save)


if __name__=='__main__':
    parser = ArgumentParser('Simulation Chamber for Dominion')
    parser.add_argument('--split', default=0, type=int, help='Starting Copper/Estate split. 0: Random, 1: 25Split, 2: 34Split')
    parser.add_argument('--strategy', default='Random', type=str, help='Strategy of AI opponent. Supported: [Random]')
    parser.add_argument('--players', default=2, type=int, help='Number of AI players')
    parser.add_argument('--prosperity', action='store_true', help='Whether the Prosperity settings should be used')
    parser.add_argument('--iters', type=int,  required=True, help='Number of games to simulate')
    parser.add_argument('--save', action='store_true', help='Whether the data should be saved')

    args = parser.parse_args()
    main(args)
