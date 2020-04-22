from typing import List

import numpy as np

from actioncard import *
from card import *
from card import Card
from cursecard import *
from treasurecard import *
from victorycard import *


def getBaseKingdomCards() -> List:
    '''Return a list of the Base set kingdom cards'''
    # TODO: Implement Sentry, Bandit, Vassal, Merchant
    return [Cellar, Chapel, Moat, \
        Harbinger, Merchant, Village, Workshop, \
        Bureaucrat, Gardens, Militia, Moneylender, Poacher, Remodel, Smithy, ThroneRoom, \
        CouncilRoom, Festival, Laboratory, Library, Market, Mine, Witch, \
        Artisan]

def containsCard(card: Card, cards: List[Card]):
    '''Returns if cards contains any instance of card.'''
    cardName = str(card)
    return any(str(c) == cardName for c in cards)

def removeFirstCard(card: Card, cards: List[Card]):
    '''Remove and return first occurrence of card in cards if it exists, else return None'''
    cardName = str(card)
    for i, c in enumerate(cards):
        if cardName == str(c):
            return cards.pop(i)
    return None

def getFirstIndex(card: Card, cards: List[Card]):
    '''Return the first index of card in cards if it exists, else -1.'''
    cardName = str(card)
    for i, c in enumerate(cards):
        if cardName == str(c):
            return i
    return -1

def running_mean(x: List, N: int):
    '''Calculate a running mean of x with window N, including the first N-1 partial averages.'''
    cumsum = np.cumsum(np.insert(x, 0, 0))
    last_N = (cumsum[N:] - cumsum[:-N]) / float(N)
    first_N = np.cumsum(x[:N-1])
    for i in range(N-1):
        first_N[i] /= (i+1)
    y = np.concatenate((first_N, last_N))
    return y
