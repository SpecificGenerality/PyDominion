import logging
import random
import sys
from abc import ABC, abstractmethod

import numpy as np

from actioncard import ActionCard
from cursecard import *
from enums import *
from heuristics import *
from mcts import *
from playerstate import PlayerState
from state import DecisionResponse, State
from rollout import *

# feature decks as counts of each card, least squares regress each against scores + offset
# try random + greedy

class Player(ABC):
    @abstractmethod
    def makeDecision(self, s: State, response: DecisionResponse):
        '''Given the current state s of the game, make a decision given the choices in s and modify response.'''
        pass

# TODO: Expand MCTS to work outside of sandbox games
class MCTSPlayer(Player):
    def __init__(self, rollout, root=Node()):
        self.root = root
        self.root.parent = self.root
        # To prevent clobbering trees loaded from file
        if not root.children:
            self.root.children = [Node(self.root) for i in range(GameConstants.StartingHands)]
        self.node = None
        self.rollout = rollout

    def get_C(self):
        '''Return time-varying C tuned for raw score reward'''
        return max(1, min(25, 25 / np.sqrt(self.root.n)))

    def reset(self, pState: PlayerState):
        self.root.n += 1
        # advance MCTS from virtual root to the correct start position (2/3/4/5 coppers)
        self.node = self.root.children[pState.getTreasureCardCount(pState.hand)-2]
        self.node.n += 1

    def get_next_node(self, choices: List[Card], C):
        '''Select the node that maximizes the UCB score'''
        max_score = 0
        next_node = None
        for c in choices:
            for node in self.node.children:
                if str(node.card) == str(c):
                    val = node.score(C)
                    if val > max_score:
                        max_score = val
                        next_node = node

        return next_node

    def makeDecision(self, s: State, response: DecisionResponse):
        player = s.decision.controllingPlayer
        d = s.decision

        if s.phase == Phase.ActionPhase:
            assert False, 'MCTS does not support action cards yet'
        elif s.phase == Phase.TreasurePhase:
            response.singleCard = d.cardChoices[0]
        else:
            if not self.node.children:
                response.singleCard = self.rollout.select([card for card in d.cardChoices if not isinstance(card, Curse)] + [None])
                return None

            self.node.add_unique_children(d.cardChoices)
            # the next node in the tree is the one that maximizes the UCB1 score
            next_node = self.get_next_node(d.cardChoices, self.get_C())
            if not next_node:
                response.singleCard = self.rollout.select([card for card in d.cardChoices if not isinstance(card, Curse)] + [None])
                return None
            response.singleCard = next_node.card
            return next_node

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
        player = d.controllingPlayer
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
        d.cardChoices.append(None)
        if s.phase == Phase.BuyPhase:
            removeFirstCard(Curse(), d.cardChoices)
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
