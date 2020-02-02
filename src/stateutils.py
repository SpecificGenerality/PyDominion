from card import Card
from enums import Zone
from victorycard import *
from typing import List
import random

def removeCard(card_list: List, card: Card) -> Card:
    for i, c in enumerate(card_list):
        if c == card:
            return card_list.pop(i)

    return None

def moveCard(src: List, dest: List, card: Card) -> None:
    x = removeCard(src, card)
    assert(x is not None)
    dest.append(card)