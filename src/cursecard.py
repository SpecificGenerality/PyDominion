from card import Card

class Curse(Card):
    def getPlusVictoryPoints(self) -> int:
        return 0

    def getVictoryPoints(self) -> int:
        return -1

    def getPlusActions(self) -> int:
        return 0

    def getPlusBuys(self) -> int:
        return 0

    def getPlusCards(self) -> int:
        return 0

    def getCoinCost(self) -> int:
        return 0

    def getPlusCoins(self) -> int:
        return 0

    def getTreasure(self):
        return 0

    def __str__(self):
        return "Curse"

    def __eq__(self, other):
        return isinstance(other, Curse)