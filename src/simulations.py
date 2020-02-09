from game import Game
from gamedata import GameData
from config import GameConfig
from enums import StartingSplit
from buyagenda import *
from victorycard import *
from player import *
import json
import time
from argparse import ArgumentParser
import numpy as np

def simulate(args: ArgumentParser, config: GameConfig, n: int):
    print('Starting game simulations...')
    start_time = time.time()
    sim_stats = {'Iters': 0, 'StartWins': 0, 'Ties': 0, 'ProvinceWins': 0, 'Degenerate': 0}
    scores = np.zeros((n, 2))
    for i in range(n):
        iter_start_time = time.time()
        print(f'====Iteration {i}====')

        data = GameData(config)

        if args.strategy == 'Random':
            playerClass = RandomPlayer
        elif args.strategy == 'BigMoney':
            playerClass = HeuristicPlayer

        players = [RandomPlayer(), HeuristicPlayer(TDBigMoneyBuyAgenda())]
        dominion = Game(config, data, players)

        dominion.newGame()
        dominion.run()
        game_stats = dominion.getStats()
        sim_stats['Iters'] += 1
        sim_stats['StartWins'] += (1 if len(game_stats['Winners']) == 1 and game_stats['Winners'][0] == 0 else 0)
        sim_stats['Ties'] += (1 if len(game_stats['Winners']) > 1 else 0)
        sim_stats['ProvinceWins'] += 1 if any(isinstance(card, Province) for card in game_stats['EmptyPiles']) else 0
        sim_stats['Degenerate'] += 1 if dominion.state.isDegenerate() else 0
        scores[i] = dominion.getPlayerScores()
        print(f'Time elapsed: {time.time() - iter_start_time}')
    with open('data/R-TDBM-1k.txt', 'w+') as file:
        json.dump(sim_stats, file)

    np.savez('data/R-TDBM-1k', scores)
    print()

    print(f'Total time elapsed: {time.time() - start_time}s')

def main(args: ArgumentParser):
    if args.split == 0:
        split = StartingSplit.StartingRandomSplit
    elif args.split == 1:
        split = StartingSplit.Starting25Split
    else:
        split = StartingSplit.Starting34Split

    config = GameConfig(split, prosperity=args.prosperity, numPlayers=args.players)

    simulate(args, config, args.iters)


if __name__=='__main__':
    parser = ArgumentParser('Simulation Chamber for Dominion')
    parser.add_argument('--split', default=0, type=int, help='Starting Copper/Estate split. 0: Random, 1: 25Split, 2: 34Split')
    parser.add_argument('--strategy', default='Random', type=str, help='Strategy of AI opponent. Supported: [Random]')
    parser.add_argument('--players', default=2, type=int, help='Number of AI players')
    parser.add_argument('--prosperity', action='store_true', help='Whether the Prosperity settings should be used')
    parser.add_argument('--iters', type=int,  required=True, help='Number of games to simulate')

    args = parser.parse_args()
    main(args)