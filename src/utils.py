from typing import List, Type, Union

import numpy as np
import torch

from card import Card

##############
# Game utils #
##############


def get_card(card: Union[Type[Card], Card], cards: List[Card]):
    '''Returns the first instance of card in cards'''
    card_type = card if isinstance(card, type) else type(card)
    for c in cards:
        if isinstance(c, card_type):
            return c
    return None


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
    # Enumerate in reverse order because it will be O(1)
    n = len(cards)
    for i in range(n):
        c = cards[n - i - 1]
        if c == card:
            return cards.pop(n - i - 1)
    return None


def move_card(card: Card, src: List, dest: List) -> None:
    x = remove_card(card, src)
    if x is None:
        raise ValueError(f'{card} not found in source list.')
    dest.append(x)

################
# Tensor utils #
################


def mov_zero(feature: torch.tensor, src: int, tgt: int, length: int) -> None:
    feature[tgt:tgt + length] = feature[tgt:tgt + length] + feature[src:src + length]
    feature[src:src + length] = 0


def dec_inc(feature: torch.tensor, src: int, tgt: int) -> None:
    feature[src] = feature[src] - 1
    feature[tgt] = feature[tgt] + 1


################
# Misc utils #
################


def running_mean(x: List, N: int):
    '''Calculate a running mean of x with window N, including the first N-1 partial averages.'''
    # Coerce the window to be valid
    N = min(len(x), max(1, N))
    cumsum = np.cumsum(np.insert(x, 0, 0))
    last_N = (cumsum[N:] - cumsum[:-N]) / float(N)
    first_N = np.cumsum(x[:N - 1])
    for i in range(N - 1):
        first_N[i] /= (i + 1)
    y = np.concatenate((first_N, last_N))
    return y
