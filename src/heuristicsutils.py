import random
from typing import List

from actioncard import ActionCard
from card import Card
from state import *
from treasurecard import TreasureCard
from victorycard import VictoryCard


def heuristic_select_cards(s: State, response: DecisionResponse, scoringFunction):
        choices = s.decision.card_choices
        random.shuffle(choices)
        choices.sort(key=scoringFunction, reverse=True)
        for i in range(max(1, s.decision.min_cards)):
            response.cards.append(choices[i])

def heuristic_best_card(choices: List[Card], scoringFunction):
        v = choices
        random.shuffle(v)
        v.sort(key=scoringFunction, reverse=True)
        return v[0]

def get_best_TD_card(choices: List[Card]):
    return max(choices, key=lambda x: x.get_plus_cards())

def has_plus_action_cards(choices: List[Card]):
    return any(card.get_plus_actions() > 0 for card in choices)

def get_first_plus_action_card(choices: List[Card]):
    return next(c.get_plus_actions() > 0 for c in choices)

def get_lowest_treasure_card(choices: List[Card]):
    return min(choices, key=lambda x: x.get_treasure())

def has_treasure_cards(choices: List[Card]):
    return any(isinstance(card, TreasureCard) for card in choices)

def get_first_no_action_card(choices: List[Card]):
    return next(c.get_plus_actions() == 0 and isinstance(c, ActionCard) for c in choices)

def get_max_plus_cards_card(choices: List[Card]):
    card = max(choices, key=lambda x: x.get_plus_cards())
    return card if card.get_plus_cards() >= 2 else None

def get_highest_VP_card(choices: List[Card]):
    return max(choices, key=lambda x: x.get_victory_points())

def has_excess_actions(choices: List[Card]):
    return sum(card.get_plus_actions() for card in choices) - sum(1 if isinstance(card, ActionCard) else 0 for card in choices)

def getExcessActionCard(choices: List[Card]):
    if has_excess_actions(choices):
        return get_first_no_action_card(choices)
    else:
        return None
