from abc import ABC, abstractmethod

class Card(ABC):
    def __init__(self, copies=1, turns_left=1):
        self.copies = copies
        self.turns_left = turns_left

    @abstractmethod
    def getCoinCost(self) -> int:
        pass

    @abstractmethod
    def getVictoryPoints(self) -> int:
        pass

    @abstractmethod
    def getPlusVictoryPoints(self) -> int:
        pass

    @abstractmethod
    def getPlusActions(self) -> int:
        pass

    @abstractmethod
    def getPlusBuys(self) -> int:
        pass

    @abstractmethod
    def getPlusCards(self) -> int:
        pass

    @abstractmethod
    def getPlusCoins(self) -> int:
        pass

    @abstractmethod
    def getTreasure(self) -> int:
        pass

    def __hash__(self) -> str:
        return hash(str(self))

    def __repr__(self) -> str:
        return str(self)