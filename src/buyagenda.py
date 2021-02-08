from abc import ABC, abstractmethod
from typing import List

from actioncard import Chapel, Witch
from buyheuristics import (big_money_buy, chapel_buy, greedy_score_function,
                           terminal_draw_buy)
from card import Card
from heuristicsutils import heuristic_best_card
from state import State
from utils import get_card
from victorycard import Province


class BuyAgenda(ABC):
    @abstractmethod
    def buy(self, s: State, player: int, choices: List[Card]):
        pass

    @abstractmethod
    def forceBuy(self, s: State, player: int, choices: List[Card]):
        pass


# implements the Big Money Optimized Buy Strategy: http://wiki.dominionstrategy.com/index.php/Big_money
class BigMoneyBuyAgenda(BuyAgenda):
    def buy(self, s: State, player: int, choices: List[Card]):
        coins = s.player_states[player].coins
        return big_money_buy(coins, choices, s.supply[Province])

    def forceBuy(self, s: State, player: int, choices: List[Card]):
        card = self.buy(s, player, choices)
        if card:
            return card

        return heuristic_best_card(choices, greedy_score_function)


class TDBigMoneyBuyAgenda(BuyAgenda):
    def buy(self, s: State, player: int, choices: List[Card]):
        coins = s.player_states[player].coins

        card = terminal_draw_buy(s.get_terminal_draw_density(player), choices)

        if card:
            return card

        return big_money_buy(coins, choices, s.supply[Province])

    def forceBuy(self, s: State, player: int, choices: List[Card]):
        card = self.buy(s, player, choices)
        if card:
            return card

        return heuristic_best_card(choices, greedy_score_function)


# http://wiki.dominionstrategy.com/index.php/Big_Money#Terminal_draw_Big_Money
class TDEBigMoneyBuyAgenda(BuyAgenda):
    def buy(self, s: State, player: int, choices: List[Card]):
        coins = s.player_states[player].coins

        # buy one chapel if we don't have one
        total_coins = s.get_total_coin_count(player)
        has_chapel = s.has_card(player, Chapel)

        card = chapel_buy(total_coins, has_chapel, choices)

        if card:
            return card

        card = terminal_draw_buy(s.get_terminal_draw_density(player), choices)

        if card:
            return card

        return big_money_buy(coins, choices, s.supply[Province])

    def forceBuy(self, s: State, player: int, choices: List[Card]):
        card = self.buy(s, player, choices)
        if card:
            return card

        return heuristic_best_card(choices, greedy_score_function)


class DoubleWitchBuyAgenda(BuyAgenda):
    def buy(self, s: State, player: int, choices: List[Card]):
        num_witches = s.get_card_count(player, Witch)

        if num_witches < 2:
            card = get_card(Witch, choices)
            if card:
                return card

        num_coins = s.get_total_coin_count(player)
        return big_money_buy(num_coins, choices, s.supply[Province])

    def forceBuy(self, s: State, player: int, choices: List[Card]):
        card = self.buy(s, player, choices)
        if card:
            return card

        return heuristic_best_card(choices, greedy_score_function)
