import random
from typing import List

from card import Card
from enums import Zone
from victorycard import *


def removeCard(card: Card, card_list: List) -> Card:
    for i, c in enumerate(card_list):
        if c == card:
            return card_list.pop(i)
    return None

def moveCard(card: Card, src: List, dest: List) -> None:
    x = removeCard(card, src)
    assert x is not None, f'Failed to remove {card} from {src}'
    dest.append(x)
