import pygame
from game import Game
from gamedata import GameData
from config import GameConfig
from enums import StartingSplit
from player import *

def main():
    config = GameConfig(StartingSplit.StartingRandomSplit, prosperity=False, numPlayers=2)
    data = GameData(config)
    players = [HumanPlayer() for i in range(2)]
    dominion = Game(config, data, players)

    dominion.newGame()
    dominion.run()

if __name__ == '__main__':
    main()
