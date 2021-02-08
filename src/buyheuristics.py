from typing import List, Type, Union

from actioncard import Chapel
from card import Card
from heuristicsutils import get_highest_VP_card, get_max_plus_cards_card
from treasurecard import Gold, Silver
from utils import get_card
from victorycard import Estate, Province, VictoryCard

TD_DENSITY = 0.1


def big_money_buy(coins: int, choices: List[Card], remaining_provinces: int):
    if coins >= 8:
        # Province is guaranteed to exist while the game isn't over
        return get_card(Province, choices)
    elif coins == 6 or coins == 7:
        card = get_card(Gold, choices)
        if not card:
            # if the Golds ran out, then the game is probably almost over
            return get_highest_VP_card(choices)
        else:
            return card
    elif coins == 5:
        card = get_card(Silver, choices)
        if remaining_provinces <= 4 or not card:
            return get_highest_VP_card(choices)
        else:
            return card
    elif coins == 3 or coins == 4:
        card = get_card(Silver, choices)
        if not card:
            return get_highest_VP_card(choices)
        else:
            return card
    elif coins == 2 and remaining_provinces <= 3:
        return get_card(Estate, choices)
    return None


def terminal_draw_buy(terminal_draw_density: float, choices: List[Card]):
    if terminal_draw_density < TD_DENSITY:
        card = get_max_plus_cards_card(choices)
        if card and card.get_plus_cards() >= 2 and card.get_plus_actions() == 0:
            return card


def chapel_buy(total_coins: int, has_chapel: bool, choices: List[Card]):
    if total_coins <= 3 or has_chapel:
        return None

    return get_card(Chapel, choices)


def greedy_score_function(card: Union[Type[Card], Card]):
    score = card.get_coin_cost()
    if isinstance(card, VictoryCard) or issubclass(card, VictoryCard):
        score -= 0.5
    return score
