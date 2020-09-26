from abc import ABC, abstractmethod
from typing import List

from actioncard import Chapel
from card import Card
from heuristicsutils import (get_highest_VP_card, get_max_plus_cards_card,
                             heuristic_best_card)
from playerstate import PlayerState
from state import State
from treasurecard import Gold, Silver
from utils import get_first_index
from victorycard import Estate, Province, VictoryCard

TD_DENSITY = 0.1

class BuyAgenda(ABC):
    @abstractmethod
    def buy(self, s: State, player: int, choices: List[Card]):
        pass

    @abstractmethod
    def forceBuy(self, s: State, player: int, choices: List[Card]):
        pass

# TODO: Do we need this?
class MCTSBuyAgenda(BuyAgenda):
    def buy(self, s: State, player: int, choices: List[Card]):
        return super().buy(s, player, choices)

    def forceBuy(self, s, player, choices):
        return super().forceBuy(s, player, choices)

class TDEBigMoneyBuyAgenda(BuyAgenda):
    def buy(self, s: State, player: int, choices: List[Card]):
        p_state: PlayerState = s.player_states[player]
        coins = s.player_states[player].coins
        cardIdx = -1

        if p_state.get_terminal_draw_density() < TD_DENSITY:
            card = get_max_plus_cards_card(choices)
            # print(f'TD Agenda buys {card}')
            if card:
                return card

        # buy one chapel if we don't have one
        if coins >= 2 and not p_state.hasCard(Chapel):
            cardIdx = get_first_index(Chapel(), choices)
            if cardIdx >= 0:
                return choices[cardIdx]

        if coins >= 8:
            # Province is guaranteed to exist while the game isn't over
            cardIdx = get_first_index(Province(), choices)
            return choices[cardIdx]
        elif coins == 6 or coins == 7:
            cardIdx = get_first_index(Gold(), choices)
            if cardIdx < 0:
                # if the Golds ran out, then the game is probably almost over
                return get_highest_VP_card(choices)
            else:
                return choices[cardIdx]
        elif coins == 5:
            cardIdx = get_first_index(Silver(), choices)
            if s.supply[Province] <= 4 or cardIdx < 0:
                return get_highest_VP_card(choices)
            else:
                return choices[cardIdx]
        elif coins == 3 or coins == 4:
            cardIdx = get_first_index(Silver(), choices)
            if cardIdx < 0:
                return get_highest_VP_card(choices)
            else:
                return choices[cardIdx]
        elif coins == 2 and s.supply[Province] <= 3:
            cardIdx = get_first_index(Estate(), choices)
            if cardIdx >= 0:
                return choices[cardIdx]
        return None

    def forceBuy(self, s: State, player: int, choices: List[Card]):
        card = self.buy(s, player, choices)
        if card:
            return card

        def scoringFunction(card: Card):
            score = card.get_coin_cost()
            if isinstance(card, VictoryCard):
                score -= 0.5
            return score
        return heuristic_best_card(choices, scoringFunction)

class TDBigMoneyBuyAgenda(BuyAgenda):
    def buy(self, s: State, player: int, choices: List[Card]):
        p_state: PlayerState = s.player_states[player]
        coins = s.player_states[player].coins
        cardIdx = -1

        if p_state.get_terminal_draw_density() < TD_DENSITY:
            card = get_max_plus_cards_card(choices)
            # print(f'TD Agenda buys {card}')
            if card:
                return card

        if coins >= 8:
            # Province is guaranteed to exist while the game isn't over
            cardIdx = get_first_index(Province(), choices)
            return choices[cardIdx]
        elif coins == 6 or coins == 7:
            cardIdx = get_first_index(Gold(), choices)
            if cardIdx < 0 or s.supply[Province] <= 4:
                # if the Golds ran out, then the game is probably almost over
                return get_highest_VP_card(choices)
            else:
                return choices[cardIdx]
        elif coins == 5:
            cardIdx = get_first_index(Silver(), choices)
            if s.supply[Province] <= 4 or cardIdx < 0:
                return get_highest_VP_card(choices)
            else:
                return choices[cardIdx]
        elif coins == 3 or coins == 4:
            cardIdx = get_first_index(Silver(), choices)
            if cardIdx < 0 or s.supply[Province] <= 2:
                return get_highest_VP_card(choices)
            else:
                return choices[cardIdx]
        elif coins == 2 and s.supply[Province] <= 3:
            cardIdx = get_first_index(Estate(), choices)
            if cardIdx >= 0:
                return choices[cardIdx]
        return None

    def forceBuy(self, s: State, player: int, choices: List[Card]):
        card = self.buy(s, player, choices)
        if card:
            return card

        def scoringFunction(card: Card):
            score = card.get_coin_cost()
            if isinstance(card, VictoryCard):
                score -= 0.5
            return score
        return heuristic_best_card(choices, scoringFunction)

# implements the Big Money Optimized Buy Strategy: http://wiki.dominionstrategy.com/index.php/Big_money
class BigMoneyBuyAgenda(BuyAgenda):
    def buy(self, s: State, player: int, choices: List[Card]):
        coins = s.player_states[player].coins
        cardIdx = -1
        if coins >= 8:
            # Province is guaranteed to exist while the game isn't over
            cardIdx = get_first_index(Province(), choices)
            return choices[cardIdx]
        elif coins == 6 or coins == 7:
            cardIdx = get_first_index(Gold(), choices)
            if cardIdx < 0:
                # if the Golds ran out, then the game is probably almost over
                return get_highest_VP_card(choices)
            else:
                return choices[cardIdx]
        elif coins == 5:
            cardIdx = get_first_index(Silver(), choices)
            if s.supply[Province] <= 4 or cardIdx < 0:
                return get_highest_VP_card(choices)
            else:
                return choices[cardIdx]
        elif coins == 3 or coins == 4:
            cardIdx = get_first_index(Silver(), choices)
            if cardIdx < 0:
                return get_highest_VP_card(choices)
            else:
                return choices[cardIdx]
        elif coins == 2 and s.supply[Province] <= 3:
            cardIdx = get_first_index(Estate(), choices)
            if cardIdx >= 0:
                return choices[cardIdx]
        return None

    def forceBuy(self, s: State, player: int, choices: List[Card]):
        card = self.buy(s, player, choices)
        if card:
            return card

        def scoringFunction(card: Card):
            score = card.get_coin_cost()
            if isinstance(card, VictoryCard):
                score -= 0.5
            return score
        return heuristic_best_card(choices, scoringFunction)
