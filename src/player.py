from abc import ABC, abstractmethod
from playerstate import PlayerState
from state import State, DecisionResponse
from enums import *
from actioncard import ActionCard
from heuristics import *
from mcts import *
from cursecard import *
import random
import logging
import sys
import numpy as np

class Player(ABC):
    @abstractmethod
    def makeDecision(self, s: State, response: DecisionResponse):
        pass


# TODO: Expand MCTS to work outside of sandbox games
class MCTSPlayer(Player):
    def __init__(self, root=Node()):
        self.root = root
        self.root.parent = self.root
        # To prevent clobbering trees loaded from file
        if not root.children:
            self.root.children = [Node(self.root) for i in range(4)]
        self.node = None
        # K: card name, V: (avg value, times played)
        self.mast = {}
        self.tau = 0.05

        # static C
        # self.C = np.sqrt(2)

    def update_mast(self, cards: List[Card], score: int):
        '''Update history heuristic with card buys from last rollout'''
        for c in cards:
            if str(c) in self.mast:
                n = self.mast[str(c)][1]
                x_bar = self.mast[str(c)][0]
                self.mast[str(c)] = (x_bar / (n+1) * n + score / (n+1), n+1)
            else:
                self.mast[str(c)] = (score, 1)
        # print(self.mast)

    def gibbs_sample_mast(self, choices: List[Card]) -> Card:
        # create gibbs distribution over choices
        D = [np.exp(self.mast.get(str(c), (0, 0))[0] / self.tau) for c in choices]
        D /= sum(D)
        return np.random.choice(choices, p=D)

    def get_C(self):
        # time-varying C
        return max(0.01, min(0.25, 1 / np.sqrt(self.root.n)))

    def reset(self, pState: PlayerState):
        self.root.n += 1
        # advance MCTS from virtual root to the correct start position (2/3/4/5 coppers)
        self.node = self.root.children[pState.getTreasureCardCount(pState.hand)-2]
        self.node.n += 1

    def get_next_node(self, choices: List[Card], C):
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
                # random rollout
                # response.singleCard = random.choice([card for card in d.cardChoices if not isinstance(card, Curse)] + [None])
                # gibbs history heuristic rollout
                response.singleCard = self.gibbs_sample_mast([card for card in d.cardChoices if not isinstance(card, Curse)] + [None])
                return None

            self.node.add_unique_children(d.cardChoices)
            # the next node in the tree is the one that maximizes the UCB1 score
            next_node = self.get_next_node(d.cardChoices, self.get_C())
            if not next_node:
                # response.singleCard = random.choice([card for card in d.cardChoices if not isinstance(card, Curse)] + [None])
                response.singleCard = self.gibbs_sample_mast([card for card in d.cardChoices if not isinstance(card, Curse)] + [None])
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