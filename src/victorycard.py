from abc import abstractmethod
from card import Card

class VictoryCard(Card):
    @abstractmethod
    def getVictoryPoints(self):
        pass

    def getPlusVictoryPoints(self):
        return 0

    def getPlusActions(self):
        return 0

    def getPlusBuys(self):
        return 0

    def getPlusCards(self):
        return 0

    def getPlusCoins(self):
        return 0

    def getTreasure(self):
        return 0

class Estate(VictoryCard):
    def getCoinCost(self):
        return 2

    def getVictoryPoints(self):
        return 1

    def __str__(self):
        return "Estate"

class Duchy(VictoryCard):
    def getCoinCost(self):
        return 5

    def getVictoryPoints(self):
        return 3

    def __str__(self):
        return "Duchy"

class Province(VictoryCard):
    def getCoinCost(self):
        return 8

    def getVictoryPoints(self):
        return 6

    def __str__(self):
        return "Province"

class Colony(VictoryCard):
    def getCoinCost(self):
        return 11

    def getVictoryPoints(self):
        return 10

    def __str__(self):
        return "Colony"

class Gardens(VictoryCard):
    def getCoinCost(self) -> int:
        return 4

    def getVictoryPoints(self):
        return 0

    def __str__(self):
        return "Gardens"



