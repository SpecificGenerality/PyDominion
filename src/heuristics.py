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
            if hasExcessActions(s.decision.cardChoices):
                if isinstance(card, ActionCard):
                    return 100 - card.getPlusActions()
                return -card.getCoinCost()
            elif hasTreasureCards(s.decision.choices):
                if isinstance(card, TreasureCard):
                    return 100 - card.getTreasure()
                return -card.getCoinCost()
            else:
                return -card.getCoinCost()
        heuristicSelectCards(s, response, scoringFunction)

    def makeDiscardDownDecision(self, s: State, response: DecisionResponse):
        d = s.decision
        def scoringFunction(card: Card):
            if isinstance(card, VictoryCard):
                return 20
            elif isinstance(card, Curse):
                return 19
            elif isinstance(card, Copper):
                return 18
            return -card.getCoinCost()

        heuristicSelectCards(s, response, scoringFunction)

    def makeCopyDecision(self, s: State, response: DecisionResponse):
        def scoringFunction(card: Card):
            return card.getCoinCost()
        heuristicSelectCards(s, response, scoringFunction)

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
                return -100-card.getCoinCost()
            return -card.getCoinCost()

        heuristicSelectCards(s, response, scoringFunction)

    # plays +Action first, then card with most +Card, then randomly
    def makeGreedyActionDecision(self, s: State, response: DecisionResponse):
        d = s.decision
        assert d.minCards == 0 and d.maxCards == 1, 'Invalid decisionparameters'
        def scoringFunction(card: Card):
            score = 0
            if card.getPlusActions() > 0:
                score += 100
            score += card.getCoinCost()
            return score
        heuristicSelectCards(s, response, scoringFunction)

    # plays all treasures
    def makeGreedyTreasureDecision(self, s: State, response: DecisionResponse):
        d = s.decision
        response.cards[0:0] = d.cardChoices
        print(f'{response.cards}')

    def makeBaseDecision(self, s: State, response: DecisionResponse):
        d = s.decision
        card = d.activeCard
        player = s.decision.controllingPlayer
        pState = s.playerStates[player]
        if isinstance(card, Cellar):
            l = 0
            for c in d.cardChoices:
                if isinstance(c, VictoryCard) or c.getCoinCost() < 2:
                    response.cards.append(c)
        elif isinstance(card, Chapel):
            treasureValue = s.playerStates[player].getTotalTreasureValue()
            trashCoppers = (treasureValue >= 7)
            l = 0
            for c in d.cardChoices:
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
            response.cards.append(d.cardChoices[0])
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
                    if isinstance(card, Gold) and s.data.supply[Gold] > 0:
                        return 20
                    if isinstance(card, Silver) and s.data.supply[Silver] > 0:
                        return 19
                    if isinstance(card, Copper) and s.data.supply[Copper] > 0:
                        return 18
                    return -card.getCoinCost()
                heuristicSelectCards(s, response, scoringFunction)
            else:
                response.cards.append(self.agenda.forceBuy(s, player, d.cardChoices))
        elif isinstance(card, Harbinger):
            def scoringFunction(card: Card):
                if hasExcessActions(pState.hand):
                    if isinstance(card, ActionCard):
                        return 100 + card.getCoinCost()
                    else:
                        return card.getCoinCost()
                else:
                    return card.getCoinCost()
            heuristicSelectCards(s, response, scoringFunction)
        elif isinstance(card, Artisan):
            event = s.events[-1]
            if not event.gained_card:
                response.cards.append(self.agenda.forceBuy(s, player, d.cardChoices))
            else:
                self.makePutDownOnDeckDecision(s, response)
        elif isinstance(card, Poacher):
            self.makeDiscardDownDecision(s, response)
        else:
            logging.error('Unexpected decision')
