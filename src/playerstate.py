from config import GameConfig
from enums import StartingSplit
from actioncard import ActionCard
from treasurecard import TreasureCard, Copper
from victorycard import VictoryCard, Estate
from typing import List

import random

class PlayerState:
    def __init__(self, gameConfig: GameConfig) -> None:
        self.actions = 1
        self.buys = 1
        self.coins = 0
        self.deck = []
        self.discard = []
        self.hand = []
        self.island = []
        self.playArea = []
        self.turns = 0

        if (gameConfig.startingSplit == StartingSplit.Starting34Split):
        	self.deck = [Copper() for i in range(3)] + [Estate() for i in range(2)] + [Copper() for i in range(4)] + [Estate()]
        elif (gameConfig.startingSplit == StartingSplit.Starting25Split):
        	self.deck = [Copper() for i in range(5)] + [Estate() for i in range(3)] + [Copper() for i in range(2)]
        else:
        	self.deck = [Copper() for i in range(7)] + [Estate() for i in range(3)]
        	random.shuffle(self.deck)

    def getTotalCards(self) -> int:
        return len(self.deck) + len(self.hand) + len(self.discard) + len(self.playArea) + len(self.island)

    def getActionCardCount(self, cardPile: List) -> int:
        return sum(isinstance(card, ActionCard) for card in cardPile)

    def getTreasureCardCount(self, cardPile: List) -> int:
        return sum(isinstance(card, TreasureCard) for card in cardPile)

    def getVictoryCardCount(self, cardPile: List) -> int:
        return sum(isinstance(card, VictoryCard) for card in cardPile)

    def getTotalCoinCount(self, cardPile: List) -> int:
        return sum(card.getPlusCoins() for card in cardPile)