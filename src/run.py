import pygame
from game import Game
from gamedata import GameData
from config import GameConfig
from enums import StartingSplit

def main():
    config = GameConfig(StartingSplit.StartingRandomSplit, prosperity=False, numPlayers=2)
    data = GameData(config)
    dominion = Game(config, data)

    dominion.newGame()
    dominion.run()

if __name__ == '__main__':
    main()
