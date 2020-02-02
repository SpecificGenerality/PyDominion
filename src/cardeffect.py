from abc import ABC, abstractmethod
from state import State, DecisionResponse

class CardEffect(ABC):
    def __init__(self):
        self.c = None

    @abstractmethod
    def playAction(self, s: State):
        pass

    def canProcessDecision(self) -> bool:
        return False

    def processDecision(self, s: State, response: DecisionResponse):
        print(f'Card does not support decisions')

    def victoryPoints(self, s: State, player: int):
        return 0