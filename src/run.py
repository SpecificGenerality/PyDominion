import argparse
from typing import List

from config import GameConfig
from constants import BASE_CARD_NAME
from enums import StartingSplit
from game import Game
from player import RandomPlayer, HeuristicPlayer, HumanPlayer, BigMoneyBuyAgenda


def main(strategies: List[str], must_include: List[str], prosperity: bool):
    must_include = [BASE_CARD_NAME.get(card_name) for card_name in must_include]
    config = GameConfig(StartingSplit.StartingRandomSplit, prosperity=prosperity, num_players=len(strategies), sandbox=False, must_include=must_include)

    players = []

    for strategy in strategies:
        if strategy == 'H':
            players.append(HumanPlayer())
        elif strategy == 'R':
            players.append(RandomPlayer())
        elif strategy == 'BM':
            players.append(HeuristicPlayer(BigMoneyBuyAgenda))

    dominion = Game(config, players)

    dominion.new_game()
    dominion.run()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--strategy', nargs='+', choices=['H', 'R', 'BM'], help='Strategy of players.')
    parser.add_argument('--must-include', nargs='+', help='Cards that must be in the kingdom (up to 10). See constants.py for supported cards.')
    parser.add_argument('--sandbox', action='store_true', help='When set, includes only the 7 basic kingdom supply cards.')
    args = parser.parse_args()

    main(args.strategy, args.must_include, args.sandbox)
