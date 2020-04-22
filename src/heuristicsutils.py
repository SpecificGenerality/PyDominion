import random
from typing import List

from actioncard import ActionCard
from card import Card
from state import *
from treasurecard import TreasureCard
from victorycard import VictoryCard


def heuristicSelectCards(s: State, response: DecisionResponse, scoringFunction):
        choices = s.decision.cardChoices
        random.shuffle(choices)
        choices.sort(key=scoringFunction, reverse=True)
        for i in range(max(1, s.decision.minCards)):
            response.cards.append(choices[i])

def heuristicBestCard(choices: List[Card], scoringFunction):
        v = choices
        random.shuffle(v)
        v.sort(key=scoringFunction, reverse=True)
        return v[0]

def getBestTDCard(choices: List[Card]):
    return max(choices, key=lambda x: x.getPlusCards())

def hasPlusActionCards(choices: List[Card]):
    return any(card.getPlusActions() > 0 for card in choices)

def getFirstPlusActionCard(choices: List[Card]):
    return next(c.getPlusActions() > 0 for c in choices)

def getLowestTreasureCard(choices: List[Card]):
    return min(choices, key=lambda x: x.getTreasure())

def hasTreasureCards(choices: List[Card]):
    return any(isinstance(card, TreasureCard) for card in choices)

def getFirstNoActionCard(choices: List[Card]):
    return next(c.getPlusActions() == 0 and isinstance(c, ActionCard) for c in choices)

def getMaxPlusCardsCard(choices: List[Card]):
    card = max(choices, key=lambda x: x.getPlusCards())
    return card if card.getPlusCards() >= 2 else None

def getHighestVPCard(choices: List[Card]):
    return max(choices, key=lambda x: x.getVictoryPoints())

def hasExcessActions(choices: List[Card]):
    return sum(card.getPlusActions() for card in choices) - sum(1 if isinstance(card, ActionCard) else 0 for card in choices)

def getExcessActionCard(choices: List[Card]):
    if hasExcessActions(choices):
        return getFirstNoActionCard(choices)
    else:
        return None
