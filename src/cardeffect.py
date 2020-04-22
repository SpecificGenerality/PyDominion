import logging
from abc import ABC, abstractmethod

from state import DecisionResponse, State


class CardEffect(ABC):
    def __init__(self):
        self.c = None

    @abstractmethod
    def playAction(self, s: State):
        pass

    def canProcessDecisions(self) -> bool:
        return False

    def processDecision(self, s: State, response: DecisionResponse):
        logging.warning('Card does not support decisions')

    def victoryPoints(self, s: State, player: int):
        return 0
