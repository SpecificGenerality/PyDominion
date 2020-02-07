from abc import ABC, abstractmethod
from state import *
from typing import List
from card import Card
from heuristics import *

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
        coins = s.playerStates[player].coins
        cardIdx = -1
        if coins >= 8:
            # Province is guaranteed to exist while the game isn't over
            cardIdx = getFirstIndex(Province(), choices)
            return choices[cardIdx]
        elif coins == 6 or coins == 7:
            cardIdx = getFirstIndex(Gold(), choices)
            if cardIdx < 0:
                # if the Golds ran out, then the game is probably almost over
                return getHighestVPCard(choices)
            else:
                return choices[cardIdx]
        elif coins == 5:
            cardIdx = getFirstIndex(Silver(), choices)
            if s.data.supply[Province] <= 4 or cardIdx < 0:
                return getHighestVPCard(choices)
            else:
                return choices[cardIdx]
        elif coins == 3 or coins == 4:
            cardIdx = getFirstIndex(Silver(), choices)
            if cardIdx < 0:
                return getHighestVPCard(choices)
            else:
                return choices[cardIdx]
        elif coins == 2 and s.data.supply[Province] <= 3:
            cardIdx = getFirstIndex(Estate(), choices)
            if cardIdx >= 0:
                return choices[cardIdx]
        return None

    def forceBuy(self, s: State, player: int, choices: List[Card]):
        card = self.buy(s, player, choices)
        if card:
            return card

        def scoringFunction(card: Card):
            score = card.getCoinCost()
            if isinstance(card, VictoryCard):
                score -= 0.5
            return score
        return heuristicBestCard(choices, scoringFunction)
