import logging

from actioncard import (ActionCard, Artisan, Bureaucrat, Cellar, Chapel,
                        Harbinger, Library, Militia, Mine, Moat, Poacher,
                        ThroneRoom)
from buyagenda import BuyAgenda
from card import Card
from cursecard import Curse
from heuristicsutils import (has_excess_actions, has_treasure_cards,
                             heuristic_select_cards, is_cantrip)
from playerstate import PlayerState
from state import DecisionResponse, DecisionState, State
from treasurecard import Copper, Gold, Silver, TreasureCard
from victorycard import Estate, VictoryCard


class PlayerHeuristic:
    def __init__(self, agenda: BuyAgenda):
        self.agenda = agenda

    def makePutDownOnDeckDecision(self, s: State, response: DecisionResponse):
        d: DecisionState = s.decision

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

        response.cards = heuristic_select_cards(d.card_choices, d.min_cards, scoringFunction)

    def makeDiscardDownDecision(self, s: State, response: DecisionResponse):
        d: DecisionState = s.decision

        def scoringFunction(card: Card):
            if isinstance(card, VictoryCard):
                return 20
            elif isinstance(card, Curse):
                return 19
            elif isinstance(card, Copper):
                return 18
            return -card.get_coin_cost()

        response.cards = heuristic_select_cards(d.card_choices, d.min_cards, scoringFunction)

    def makeCopyDecision(self, s: State, response: DecisionResponse):
        d: DecisionState = s.decision

        def scoringFunction(card: Card):
            return card.get_coin_cost()

        response.cards = heuristic_select_cards(d.card_choices, d.min_cards, scoringFunction)

    def makeTrashDecision(self, s: State, response: DecisionResponse):
        d: DecisionState = s.decision

        def scoringFunction(card: Card):
            if isinstance(card, Curse):
                return 20
            elif isinstance(card, Estate):
                return 19
            elif isinstance(card, Copper):
                return 18
            elif isinstance(card, VictoryCard):
                return -100 - card.get_coin_cost()
            return -card.get_coin_cost()

        response.cards = heuristic_select_cards(d.card_choices, d.min_cards, scoringFunction)

    # plays +Action first, then card with most +Card, then randomly
    def makeGreedyActionDecision(self, s: State, response: DecisionResponse):
        d: DecisionState = s.decision
        assert d.min_cards == 0 and d.max_cards == 1, 'Invalid decision parameters'

        def scoringFunction(card: Card):
            '''Play all cantrips first, then greedily'''
            cantrip_bonus = 1
            score = min(card.get_coin_cost(), 6)

            if is_cantrip(card):
                score += cantrip_bonus

            return score

        cards = heuristic_select_cards(d.card_choices, d.min_cards, scoringFunction)
        response.cards = cards

    def makeBaseDecision(self, s: State, response: DecisionResponse):
        d: DecisionState = s.decision
        card = d.active_card
        player = s.decision.controlling_player
        p_state: PlayerState = s.player_states[player]
        if isinstance(card, Cellar):
            num_discarded = 0
            for c in d.card_choices:
                if isinstance(c, VictoryCard) or c.get_coin_cost() < 2:
                    response.cards.append(c)
        elif isinstance(card, Chapel):
            treasureValue = s.get_total_coin_count(player)
            trashCoppers = (treasureValue > 3)
            num_discarded = 0
            for c in d.card_choices:
                if num_discarded == 4:
                    break
                if isinstance(c, Curse):
                    response.cards.append(c)
                    num_discarded += 1
                elif isinstance(c, Copper) and trashCoppers:
                    response.cards.append(c)
                    num_discarded += 1
                elif isinstance(c, Estate):
                    response.cards.append(c)
                    num_discarded += 1
                elif isinstance(c, Chapel):
                    response.cards.append(c)
                    num_discarded += 1
        elif isinstance(card, Moat):
            response.choice = 0
        elif isinstance(card, Bureaucrat):
            response.cards.append(d.card_choices[0])
        elif isinstance(card, Militia):
            self.makeDiscardDownDecision(s, response)
        elif isinstance(card, ThroneRoom):
            self.makeCopyDecision(s, response)
        elif isinstance(card, Library):
            if s.player_states[s.player].actions == 0:
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
                response.cards = heuristic_select_cards(d.card_choices, d.min_cards, scoringFunction)
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
            response.cards = heuristic_select_cards(d.card_choices, d.min_cards, scoringFunction)
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
