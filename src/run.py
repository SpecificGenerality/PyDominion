from config import GameConfig
from enums import StartingSplit
from game import Game
from supply import Supply
from player import *


def main():
    config = GameConfig(StartingSplit.StartingRandomSplit, prosperity=False, num_players=2, sandbox=True)
    data = Supply(config)
    players = [HumanPlayer(), HeuristicPlayer(BigMoneyBuyAgenda())]
    dominion = Game(config, data, players)

    dominion.newGame()
    dominion.run()

if __name__ == '__main__':
    main()
