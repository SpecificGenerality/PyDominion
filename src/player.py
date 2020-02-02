from abc import ABC, abstractmethod
from playerstate import PlayerState
from state import State, DecisionResponse
from enums import *

class Player(ABC):
    def __init__(self, player_id: int, s: PlayerState):
        self.id = player_id
        self.pState = s
        return

    @abstractmethod
    def makeDecision(self, s: State, response: DecisionResponse):
        pass

class HumanPlayer(Player):
    def makeDecision(self, s: State, response: DecisionResponse):
        d = s.decision
        if d.type == DecisionType.DecisionSelectCards:
            cardsToPick = -1
            d.printCardChoices()
            while (cardsToPick < d.minCards or cardsToPick > d.maxCards):
                cardsToPick = int(input(f'Pick between {d.minCards} and {d.maxCards} of the above cards:\n'))

            responseIdxs = []
            for i in range(cardsToPick):
                cardIdx = -1
                while (cardIdx == -1 or cardIdx in responseIdxs):
                    d.printCardChoices()
                    cardIdx = int(input(f'Choose another card:\n'))
                responseIdxs.append(cardIdx)
                response.cards.append(d.cardChoices[cardIdx])
        elif d.type == DecisionType.DecisionDiscreteChoice:
            choice = -1
            while choice == -1 or choice > d.minCards:
                choice = int(input(f'Please make a discrete choice from the above cards:\n'))
                d.printCardChoices()
            response.choice = choice
        else:
            print(f'Player {self.id} given invalid decision type.')

    def __str__(self):
        return f"Human Player {self.id}"
