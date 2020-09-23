from typing import List

import numpy as np

from card import Card


def get_first_index(card: Card, cards: List[Card]):
    '''Return the first index of card in cards if it exists, else -1.'''
    cardName = str(card)
    for i, c in enumerate(cards):
        if cardName == str(c):
            return i
    return -1

def contains_card(card: Card, cards: List[Card]):
    '''Returns if cards contains any instance of card.'''
    cardName = str(card)
    return any(str(c) == cardName for c in cards)

def remove_first_card(card: Card, cards: List[Card]):
    '''Remove and return first occurrence of card in cards if it exists, else return None'''
    cardName = str(card)
    for i, c in enumerate(cards):
        if cardName == str(c):
            return cards.pop(i)
    return None

def remove_card(card: Card, cards: List) -> Card:
    for i, c in enumerate(cards):
        if c == card:
            return cards.pop(i)
    return None

def move_card(card: Card, src: List, dest: List) -> None:
    x = remove_card(card, src)
    if x is None:
        raise ValueError(f'{card} not found in source list.')
    dest.append(x)

def running_mean(x: List, N: int):
    '''Calculate a running mean of x with window N, including the first N-1 partial averages.'''
    # Coerce the window to be valid
    N = min(len(x), max(1, N))
    cumsum = np.cumsum(np.insert(x, 0, 0))
    last_N = (cumsum[N:] - cumsum[:-N]) / float(N)
    first_N = np.cumsum(x[:N-1])
    for i in range(N-1):
        first_N[i] /= (i+1)
    y = np.concatenate((first_N, last_N))
    return y
