from abc import ABC, abstractmethod
from playerstate import PlayerState
from state import State, DecisionResponse
from enums import *
from actioncard import ActionCard
from heuristics import *
import random
import logging

class Player(ABC):
    @abstractmethod
    def makeDecision(self, s: State, response: DecisionResponse):
        pass

class HeuristicPlayer(Player):
    def __init__(self, agenda: BuyAgenda):
        self.heuristic = PlayerHeuristic(agenda)

    def makePhaseDecision(self, s: State, response: DecisionResponse):
        player = s.decision.controllingPlayer
        d = s.decision
        if s.phase == Phase.ActionPhase:
            self.heuristic.makeGreedyActionDecision(s, response)
        elif s.phase == Phase.TreasurePhase:
            response.singleCard = d.cardChoices[0]
        else:
            response.singleCard = self.heuristic.agenda.buy(s, player, d.cardChoices)
        return

    def makeDecision(self, s: State, response: DecisionResponse):
        d = s.decision
        if d.type != DecisionType.DecisionSelectCards and d.type != DecisionType.DecisionDiscreteChoice:
            logging.error('Invalid decision type')
        if not d.activeCard:
            self.makePhaseDecision(s, response)
        elif s.events:
            event = s.events[-1]
            if isinstance(event, PutOnDeckDownToN):
                self.heuristic.makePutDownOnDeckDecision(s, response)
            elif isinstance(event, DiscardDownToN):
                self.heuristic.makeDiscardDownDecision(s, response)
            elif isinstance(event, RemodelExpand):
                if not event.trashed_card:
                    def scoringFunction(card: Card):
                        if isinstance(card, Curse):
                            return 19
                        elif isinstance(card, Estate):
                            return 18
                        elif isinstance(card, VictoryCard):
                            return -200 + card.getCoinCost()
                        return -card.getCoinCost()
                    heuristicSelectCards(s, response, scoringFunction)
                else:
                    response.cards.append(self.heuristic.agenda.forceBuy(s, player, d.cardChoices))
            else:
                self.heuristic.makeBaseDecision(s, response)

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
            logging.error(f'Invalid decision type')

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
            logging.error(f'Player {self.id} given invalid decision type.')

    def __str__(self):
        return f"Human Player"


class PlayerInfo:
    def __init__(self, id: int, controller: Player):
        self.id = id
        self.controller = controller

    def __str__(self):
        return f'{self.controller} {self.id}'