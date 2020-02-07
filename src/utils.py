from card import *
from treasurecard import *
from cursecard import *
from victorycard import *
from actioncard import *
from typing import List
from card import Card

def getBaseKingdomCards() -> List:
    # TODO: Implement Sentry, Bandit, Vassal, Merchant
    return [Cellar, Chapel, Moat, \
        Harbinger, Merchant, Village, Workshop, \
        Bureaucrat, Gardens, Militia, Moneylender, Poacher, Remodel, Smithy, ThroneRoom, \
        CouncilRoom, Festival, Laboratory, Library, Market, Mine, Witch, \
        Artisan]

def containsCard(card: Card, cards: List[Card]):
    cardName = str(card)
    for i, c in enumerate(cards):
        if cardName == str(c):
            return True
    return False

def removeFirstCard(card: Card, cards: List[Card]):
    cardName = str(card)
    for i, c in enumerate(cards):
        if cardName == str(c):
            return cards.pop(i)
    return None

def getFirstIndex(card: Card, cards: List[Card]):
    cardName = str(card)
    for i, c in enumerate(cards):
        if cardName == str(c):
            return i
    return -1
