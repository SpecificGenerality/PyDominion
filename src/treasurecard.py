from abc import ABC
from card import Card

class TreasureCard(Card):
    def getPlusVictoryPoints(self) -> int:
        return 0

    def getVictoryPoints(self) -> int:
        return 0

    def getPlusActions(self) -> int:
        return 0

    def getPlusBuys(self) -> int:
        return 0

    def getPlusCards(self) -> int:
        return 0

class Copper(TreasureCard):
    def getCoinCost(self) -> int:
        return 0

    def getPlusCoins(self) -> int:
        return 1

    def getTreasure(self):
        return 1

    def __str__(self):
        return "Copper"

class Silver(TreasureCard):
    def getCoinCost(self) -> int:
        return 3

    def getPlusCoins(self) -> int:
        return 2

    def getTreasure(self):
        return 2

    def __str__(self):
        return "Silver"

class Gold(TreasureCard):
    def getCoinCost(self) -> int:
        return 6

    def getPlusCoins(self) -> int:
        return 3

    def getTreasure(self):
        return 3

    def __str__(self):
        return "Gold"

class Platinum(TreasureCard):
    def getCoinCost(self) -> int:
        return 9

    def getPlusCoins(self) -> int:
        return 5

    def getTreasure(self):
        return 5

    def __str__(self):
        return "Platinum"