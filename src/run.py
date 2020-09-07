import argparse
from config import GameConfig
from enums import StartingSplit
from game import Game
from supply import Supply
from player import *


def main(prosperity: bool):
    config = GameConfig(StartingSplit.StartingRandomSplit, prosperity=prosperity, num_players=2, sandbox=False)
    data = Supply(config)
    players = [HumanPlayer(), HeuristicPlayer(BigMoneyBuyAgenda())]
    dominion = Game(config, data, players)

    dominion.newGame()
    dominion.run()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sandbox', action='store_true', help='When set, includes only the 7 basic kingdom supply cards.')
    args = parser.parse_args()

    main(args.sandbox)
