import random
from typing import Callable, List

from actioncard import ActionCard
from card import Card
from treasurecard import TreasureCard
from victorycard import VictoryCard


def heuristic_select_cards(choices: List[Card], k: int, scoringFunction: Callable[[Card], float]) -> List[Card]:
        random.shuffle(choices)
        choices.sort(key=scoringFunction, reverse=True)
        cards = []
        for i in range(max(1, k+1)):
            cards.append(choices[i])
        return cards

def heuristic_best_card(choices: List[Card], scoringFunction: Callable[[Card], float]) -> Card:
        v = choices
        random.shuffle(v)
        v.sort(key=scoringFunction, reverse=True)
        return v[0]

def get_best_TD_card(choices: List[Card]) -> Card:
    return max(choices, key=lambda x: x.get_plus_cards())

def has_plus_action_cards(choices: List[Card]) -> bool:
    return any(card.get_plus_actions() > 0 for card in choices)

def get_first_plus_action_card(choices: List[Card]) -> Card:
    try:
        return next(filter(lambda x: x.get_plus_actions() > 0, choices))
    except StopIteration:
        return None

def get_lowest_treasure_card(choices: List[Card]) -> Card:
    try:
        return min(filter(lambda x: isinstance(x, TreasureCard), choices), key=lambda x: x.get_treasure())
    except ValueError:
        return None

def has_treasure_cards(choices: List[Card]) -> Card:
    return any(isinstance(card, TreasureCard) for card in choices)

def get_first_no_action_card(choices: List[Card]) -> Card:
    try:
        return next(filter(lambda x: x.get_plus_actions() == 0 and isinstance(x, ActionCard), choices))
    except StopIteration:
        return None

def get_max_plus_cards_card(choices: List[Card]) -> Card:
    card = max(choices, key=lambda x: x.get_plus_cards())
    return card if card.get_plus_cards() >= 2 else None

def get_highest_VP_card(choices: List[Card]) -> Card:
    return max(choices, key=lambda x: x.get_victory_points())

def has_excess_actions(choices: List[Card]) -> bool:
    action_card_iter = filter(lambda x: isinstance(x, ActionCard), choices)
    return sum(x.get_plus_actions() - 1 for x in action_card_iter) > 0
