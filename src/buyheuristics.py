from typing import List

from actioncard import Chapel
from card import Card
from heuristicsutils import get_highest_VP_card, get_max_plus_cards_card
from treasurecard import Gold, Silver
from utils import get_first_index
from victorycard import Estate, Province

TD_DENSITY = 0.1


def big_money_buy(coins: int, choices: List[Card], remaining_provinces: int):
    card_idx = -1
    if coins >= 8:
        # Province is guaranteed to exist while the game isn't over
        card_idx = get_first_index(Province(), choices)
        return choices[card_idx]
    elif coins == 6 or coins == 7:
        card_idx = get_first_index(Gold(), choices)
        if card_idx < 0:
            # if the Golds ran out, then the game is probably almost over
            return get_highest_VP_card(choices)
        else:
            return choices[card_idx]
    elif coins == 5:
        card_idx = get_first_index(Silver(), choices)
        if remaining_provinces <= 4 or card_idx < 0:
            return get_highest_VP_card(choices)
        else:
            return choices[card_idx]
    elif coins == 3 or coins == 4:
        card_idx = get_first_index(Silver(), choices)
        if card_idx < 0:
            return get_highest_VP_card(choices)
        else:
            return choices[card_idx]
    elif coins == 2 and remaining_provinces <= 3:
        card_idx = get_first_index(Estate(), choices)
        if card_idx >= 0:
            return choices[card_idx]
    return None


def terminal_draw_buy(terminal_draw_density: float, choices: List[Card]):
    if terminal_draw_density < TD_DENSITY:
        card = get_max_plus_cards_card(choices)
        if card and card.get_plus_cards() >= 2 and card.get_plus_actions() == 0:
            return card


def chapel_buy(total_coins: int, has_chapel: bool, choices: List[Card]):
    if total_coins <= 3 or has_chapel:
        return None

    card_idx = get_first_index(Chapel(), choices)
    if card_idx >= 0:
        return choices[card_idx]
