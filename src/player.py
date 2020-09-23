import logging
import random
import sys
from abc import ABC, abstractmethod
from typing import List

import numpy as np

from actioncard import ActionCard
from buyagenda import BuyAgenda
from card import Card
from cursecard import Curse
from enums import *
from heuristics import PlayerHeuristic
from mcts import Node
from playerstate import PlayerState
from rollout import (HistoryHeuristicRollout, LinearRegressionRollout,
                     RandomRollout)
from state import DecisionResponse, DecisionState, State
from utils import remove_first_card

# feature decks as counts of each card, least squares regress each against scores + offset
# try random + greedy

class Player(ABC):
    @abstractmethod
    def makeDecision(self, s: State, response: DecisionResponse):
        '''Given the current state s of the game, make a decision given the choices in s and modify response.'''
        pass

# TODO: Expand MCTS to work outside of sandbox games
class MCTSPlayer(Player):
    def __init__(self, rollout, root=Node(), train=False, C=lambda x: max(1, min(25, 25 / np.sqrt(x)))):
        self.train = train
        self.root = root
        self.root.parent = self.root
        # To prevent clobbering trees loaded from file
        if not root.children:
            self.root.children = [Node(parent=self.root) for i in range(GameConstants.StartingHands)]
        self.node = None
        self.rollout = rollout
        self.Cfx = C

    def get_C(self):
        '''Return time-varying C tuned for raw score reward'''
        return self.Cfx(self.node.n)

    def reset(self, p_state: PlayerState):
        if self.train:
            self.root.n += 1
        # advance MCTS from virtual root to the correct start position (2/3/4/5 coppers)
        self.node = self.root.children[p_state.get_treasure_card_count(Zone.Hand)-2]
        if self.train:
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
        player = s.decision.controlling_player
        d: DecisionState = s.decision
        if s.phase == Phase.ActionPhase:
            assert False, 'MCTS does not support action cards yet'
        elif s.phase == Phase.TreasurePhase:
            response.single_card = d.card_choices[0]
        else:
            choices = list(filter(lambda x: not isinstance(x, Curse), d.card_choices + [None]))
            if not self.node.children:
                response.single_card = self.rollout.select(choices)
                return None

            if self.train:
                self.node.add_unique_children(d.card_choices)
            # the next node in the tree is the one that maximizes the UCB1 score
            next_node = self.get_next_node(d.card_choices, self.get_C())
            if not next_node:
                response.single_card = self.rollout.select(choices)
                return None

            self.node = next_node
            response.single_card = next_node.card
            return next_node

class HeuristicPlayer(Player):
    def __init__(self, agenda: BuyAgenda):
        self.heuristic = PlayerHeuristic(agenda)

    def makePhaseDecision(self, s: State, response: DecisionResponse):
        d: DecisionState = s.decision
        player = d.controlling_player
        if s.phase == Phase.ActionPhase:
            self.heuristic.makeGreedyActionDecision(s, response)
        elif s.phase == Phase.TreasurePhase:
            response.single_card = d.card_choices[0]
        else:
            response.single_card = self.heuristic.agenda.buy(s, player, d.card_choices)
        return

    def makeDecision(self, s: State, response: DecisionResponse):
        d: DecisionState = s.decision
        player = d.controlling_player
        if d.type != DecisionType.DecisionSelectCards and d.type != DecisionType.DecisionDiscreteChoice:
            logging.error('Invalid decision type')
        if not d.active_card:
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
                            return -200 + card.get_coin_cost()
                        return -card.get_coin_cost()
                    response.cards = heuristic_select_cards(d.card_choices, d.min_cards, scoringFunction)
                else:
                    response.cards.append(self.heuristic.agenda.forceBuy(s, player, d.card_choices))
            else:
                self.heuristic.makeBaseDecision(s, response)

class RandomPlayer(Player):
    def makeDecision(self, s: State, response: DecisionResponse):
        d: DecisionState = s.decision

        # Do not allow RandomPlayer to purchase curses
        if s.phase == Phase.BuyPhase:
            remove_first_card(Curse(), d.card_choices)

        if d.type == DecisionType.DecisionSelectCards:
            cards_to_pick = d.min_cards
            if d.max_cards > d.min_cards:
                cards_to_pick = random.randint(d.min_cards, d.max_cards)

            response.cards = random.sample(d.card_choices, k=min(cards_to_pick, len(d.card_choices)))
        elif d.type == DecisionType.DecisionDiscreteChoice:
            response.choice = random.randint(0, d.min_cards)
        else:
            logging.error(f'Invalid decision type')

    def __str__(self):
        return f'Random Player'

class HumanPlayer(Player):
    def makeDecision(self, s: State, response: DecisionResponse):
        d: DecisionState = s.decision
        if d.type == DecisionType.DecisionSelectCards:
            cardsToPick = -1
            d.print_card_choices()
            while (cardsToPick < d.min_cards or cardsToPick > d.max_cards):
                text = ''
                while not text:
                    text = input(f'Pick between {d.min_cards} and {d.max_cards} of the above cards:\n')
                cardsToPick = int(text)

            responseIdxs = []
            for i in range(cardsToPick):
                cardIdx = -1
                while (cardIdx == -1 or cardIdx in responseIdxs or cardIdx >= len(d.card_choices)):
                    d.print_card_choices()
                    text = ''
                    while not text:
                        text = input(f'Choose another card:\n')
                    cardIdx = int(text)
                responseIdxs.append(cardIdx)
                response.cards.append(d.card_choices[cardIdx])
        elif d.type == DecisionType.DecisionDiscreteChoice:
            choice = -1
            while choice == -1 or choice > d.min_cards:
                text = ''
                while not text:
                    text = input(f'Please make a discrete choice from the above cards:\n')
                choice = int(text)
                d.print_card_choices()
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
