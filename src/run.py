import argparse
from typing import List

import torch

from buyagenda import BigMoneyBuyAgenda
from config import GameConfig
from constants import BASE_CARD_NAME, SANDBOX_CARDS
from enums import StartingSplit
from game import Game
from mlp import SandboxMLP
from player import HeuristicPlayer, HumanPlayer, MLPPlayer, RandomPlayer


def main(strategies: List[str], must_include: List[str], prosperity: bool, sandbox: bool, **kwargs):
    must_include = [BASE_CARD_NAME.get(card_name) for card_name in must_include]
    config = GameConfig(StartingSplit.StartingRandomSplit, prosperity=prosperity, num_players=len(strategies), sandbox=sandbox, must_include=must_include)

    players = []

    for strategy in strategies:
        if strategy == 'H':
            players.append(HumanPlayer())
        elif strategy == 'R':
            players.append(RandomPlayer())
        elif strategy == 'BM':
            players.append(HeuristicPlayer(BigMoneyBuyAgenda()))
        elif strategy == 'MLP':
            model = SandboxMLP(14,7,1)
            model.load_state_dict(torch.load(kwargs['path']))
            model.cuda()
            players.append(MLPPlayer(model, [card_class() for card_class in SANDBOX_CARDS], 2))

    dominion = Game(config, players)

    dominion.new_game()
    dominion.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--strategy', nargs='+', choices=['H', 'R', 'BM', 'MLP'], help='Strategy of players.')
    parser.add_argument('--path', type=str, help='path to MLP model')
    parser.add_argument('--must-include', default=[], nargs='+', help='Cards that must be in the kingdom (up to 10). See constants.py for supported cards.')
    parser.add_argument('--prosperity', action='store_true', help='When set, enables the prosperity rules.')
    parser.add_argument('--sandbox', action='store_true', help='When set, includes only the 7 basic kingdom supply cards.')
    args = parser.parse_args()

    main(args.strategy, args.must_include, args.prosperity, args.sandbox, path=args.path)
