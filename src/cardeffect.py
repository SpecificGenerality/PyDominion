import logging
from abc import ABC, abstractmethod

from state import DecisionResponse, State


class CardEffect(ABC):
    def __init__(self):
        self.c = None

    @abstractmethod
    def play_action(self, s: State):
        pass

    def can_process_decisions(self) -> bool:
        return False

    def process_decision(self, s: State, response: DecisionResponse):
        logging.warning('Card does not support decisions')

    def victory_points(self, s: State, player: int):
        return 0
