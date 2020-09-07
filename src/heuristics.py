import logging
from random import shuffle
from typing import List

from buyagenda import *
from card import Card
from heuristicsutils import *
from state import *


class PlayerHeuristic():
    def __init__(self, agenda: BuyAgenda):
        self.agenda = agenda

    def makePutDownOnDeckDecision(self, s: State, response: DecisionResponse):
        d = s.decision
        def scoringFunction(card: Card):
            if has_excess_actions(s.decision.card_choices):
                if isinstance(card, ActionCard):
                    return 100 - card.get_plus_actions()
                return -card.get_coin_cost()
            elif has_treasure_cards(s.decision.choices):
                if isinstance(card, TreasureCard):
                    return 100 - card.get_treasure()
                return -card.get_coin_cost()
            else:
                return -card.get_coin_cost()
        heuristic_select_cards(s, response, scoringFunction)

    def makeDiscardDownDecision(self, s: State, response: DecisionResponse):
        d = s.decision
        def scoringFunction(card: Card):
            if isinstance(card, VictoryCard):
                return 20
            elif isinstance(card, Curse):
                return 19
            elif isinstance(card, Copper):
                return 18
            return -card.get_coin_cost()

        heuristic_select_cards(s, response, scoringFunction)

    def makeCopyDecision(self, s: State, response: DecisionResponse):
        def scoringFunction(card: Card):
            return card.get_coin_cost()
        heuristic_select_cards(s, response, scoringFunction)

    def makeTrashDecision(self, s: State, response: DecisionResponse):
        d = s.decision
        def scoringFunction(card: Card):
            if isinstance(card, Curse):
                return 20
            elif isinstance(card, Estate):
                return 19
            elif isinstance(card, Copper):
                return 18
            elif isinstance(card, VictoryCard):
                return -100-card.get_coin_cost()
            return -card.get_coin_cost()

        heuristic_select_cards(s, response, scoringFunction)

    # plays +Action first, then card with most +Card, then randomly
    def makeGreedyActionDecision(self, s: State, response: DecisionResponse):
        d: DecisionState = s.decision
        assert d.min_cards == 0 and d.max_cards == 1, 'Invalid decisionparameters'
        def scoringFunction(card: Card):
            score = 0
            if card.get_plus_actions() > 0:
                score += 100
            score += card.get_coin_cost()
            return score
        heuristic_select_cards(s, response, scoringFunction)

    # plays all treasures
    def makeGreedyTreasureDecision(self, s: State, response: DecisionResponse):
        d: DecisionState = s.decision
        response.cards[0:0] = d.card_choices
        print(f'{response.cards}')

    def makeBaseDecision(self, s: State, response: DecisionResponse):
        d: DecisionState = s.decision
        card = d.active_card
        player = s.decision.controllingPlayer
        p_state: PlayerState = s.playerStates[player]
        if isinstance(card, Cellar):
            l = 0
            for c in d.card_choices:
                if isinstance(c, VictoryCard) or c.get_coin_cost() < 2:
                    response.cards.append(c)
        elif isinstance(card, Chapel):
            treasureValue = s.playerStates[player].getTotalTreasureValue()
            trashCoppers = (treasureValue >= 7)
            l = 0
            for c in d.card_choices:
                if l == 4:
                    break
                if isinstance(c, Copper) and trashCoppers:
                    response.cards.append(c)
                    l += 1
                elif isinstance(c, Estate):
                    response.cards.append(c)
                    l += 1
        elif isinstance(card, Moat):
            response.choice = 0
        elif isinstance(card, Bureaucrat):
            response.cards.append(d.card_choices[0])
        elif isinstance(card, Militia):
            makeDiscardDownDecision(s, response)
        elif isinstance(card, ThroneRoom):
            makeCopyDecision(s, response)
        elif isinstance(card, Library):
            if s.playerStates[s.player].actions == 0:
                response.choice = 0
            else:
                response.choice = 1
        elif isinstance(card, Mine):
            event = s.events[-1]
            if not event.trashed_card:
                def scoringFunction(card: Card):
                    if isinstance(card, Gold) and s.supply[Gold] > 0:
                        return 20
                    if isinstance(card, Silver) and s.supply[Silver] > 0:
                        return 19
                    if isinstance(card, Copper) and s.supply[Copper] > 0:
                        return 18
                    return -card.get_coin_cost()
                heuristic_select_cards(s, response, scoringFunction)
            else:
                response.cards.append(self.agenda.forceBuy(s, player, d.card_choices))
        elif isinstance(card, Harbinger):
            def scoringFunction(card: Card):
                if has_excess_actions(p_state.hand):
                    if isinstance(card, ActionCard):
                        return 100 + card.get_coin_cost()
                    else:
                        return card.get_coin_cost()
                else:
                    return card.get_coin_cost()
            heuristic_select_cards(s, response, scoringFunction)
        elif isinstance(card, Artisan):
            event = s.events[-1]
            if not event.gained_card:
                response.cards.append(self.agenda.forceBuy(s, player, d.card_choices))
            else:
                self.makePutDownOnDeckDecision(s, response)
        elif isinstance(card, Poacher):
            self.makeDiscardDownDecision(s, response)
        else:
            logging.error('Unexpected decision')
