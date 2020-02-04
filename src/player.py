from abc import ABC, abstractmethod
from playerstate import PlayerState
from state import State, DecisionResponse
from enums import *
from actioncard import ActionCard
import random

class Player(ABC):
    @abstractmethod
    def makeDecision(self, s: State, response: DecisionResponse):
        pass
class RandomPlayer(Player):
    def makeDecision(self, s: State, response: DecisionResponse):
        d = s.decision
        if d.type == DecisionType.DecisionSelectCards:
            cardsToPick = d.minCards
            if d.maxCards > d.minCards:
                cardsToPick = random.randint(d.minCards, d.maxCards)
            responseIdxs = []
            for i in range(0, cardsToPick):
                choice = random.randint(0, len(d.cardChoices)-1)
                while choice in responseIdxs:
                    choice = random.randint(0, len(d.cardChoices)-1)
                responseIdxs.append(choice)
                response.cards.append(d.cardChoices[choice])
        elif d.type == DecisionType.DecisionDiscreteChoice:
            response.choice = random.randint(0, d.minCards)
        else:
            print(f'ERROR: Invalid Decision type')

    def __str__(self):
        return f'Random Player'
class HumanPlayer(Player):
    def makeDecision(self, s: State, response: DecisionResponse):
        d = s.decision
        if d.type == DecisionType.DecisionSelectCards:
            cardsToPick = -1
            d.printCardChoices()
            while (cardsToPick < d.minCards or cardsToPick > d.maxCards):
                text = ''
                while not text:
                    text = input(f'Pick between {d.minCards} and {d.maxCards} of the above cards:\n')
                cardsToPick = int(text)

            responseIdxs = []
            for i in range(cardsToPick):
                cardIdx = -1
                while (cardIdx == -1 or cardIdx in responseIdxs or cardIdx >= len(d.cardChoices)):
                    d.printCardChoices()
                    text = ''
                    while not text:
                        text = input(f'Choose another card:\n')
                    cardIdx = int(text)
                responseIdxs.append(cardIdx)
                response.cards.append(d.cardChoices[cardIdx])
                # if len(responseIdxs) == 1 and isinstance(d.cardChoices[cardIdx], ActionCard):
                #     response.singleCard = d.cardChoices[cardIdx]
        elif d.type == DecisionType.DecisionDiscreteChoice:
            choice = -1
            while choice == -1 or choice > d.minCards:
                text = ''
                while not text:
                    text = input(f'Please make a discrete choice from the above cards:\n')
                choice = int(text)
                d.printCardChoices()
            response.choice = choice
        else:
            print(f'Player {self.id} given invalid decision type.')

    def __str__(self):
        return f"Human Player"


class PlayerInfo:
    def __init__(self, id: int, controller: Player):
        self.id = id
        self.controller = controller

    def __str__(self):
        return f'{self.controller} {self.id}'