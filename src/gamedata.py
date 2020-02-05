from config import GameConfig
from typing import List, Dict
from treasurecard import *
from victorycard import *
from cursecard import *
from enums import *
from actioncard import *
from random import shuffle

class GameData:

    def __init__(self, config: GameConfig):
        def initKingdomCards(supply: Dict, must_include = [Poacher]) -> None:
            for i in range(min(config.kingdomSize, len(must_include))):
                supply[must_include[i]] = 10

            shuffle(config.randomizerDeck)
            for i in range(config.kingdomSize - len(must_include)):
                supply[config.randomizerDeck[i]] = 10

        def initSupply(supply: Dict) -> None:
            if config.numPlayers == 2:
                supply[Copper] = 46
                supply[Curse] = 0
                supply[Estate] = 8
                supply[Duchy]= 8
                supply[Province] = 8
            elif config.numPlayers == 3:
                supply[Copper] = 39
                supply[Curse] = 20
                supply[Estate] = 12
                supply[Duchy]= 12
                supply[Province] = 12
            else:
                supply[Copper] = 32
                supply[Curse] = 30
                supply[Estate] = 12
                supply[Duchy]= 12
                supply[Province] = 12
            supply[Silver] = 40
            supply[Gold] = 30

        self.supply = {}
        initSupply(self.supply)
        initKingdomCards(self.supply)
        print(f'Supply: {self.supply}')
        self.players = range(config.numPlayers)
        self.trash = []

if __name__=='__main__':
    x = GameData(GameConfig(StartingSplit.Starting34Split, False, 2))