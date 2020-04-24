import random
from typing import List

from actioncard import ActionCard, Merchant
from card import Card
from config import GameConfig
from enums import StartingSplit
from treasurecard import *
from victorycard import Estate, VictoryCard


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
        	self.deck = [Copper() for i in range(7)] + [Copper() for i in range(3)]
        	random.shuffle(self.deck)

    def getAllCards(self) -> List[Card]:
        allCards = self.hand.copy()
        allCards[0:0] = self.deck
        allCards[0:0] = self.discard
        allCards[0:0] = self.playArea
        allCards[0:0] = self.island
        return allCards

    def getTerminalActionDensity(self) -> float:
        allCards = self.getAllCards()
        return sum(1 if isinstance(card, ActionCard) and card.getPlusActions() == 0 else 0 for card in allCards) / len(allCards)

    def getTerminalDrawDensity(self) -> float:
        allCards = self.getAllCards()
        return sum(1 if isinstance(card, ActionCard) and card.getPlusActions() == 0 and card.getPlusCards() > 0 else 0 for card in allCards) / len(allCards)

    def getTotalTreasureValue(self) -> int:
        return sum(c.getTreasure() for c in self.getAllCards())

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

    def hasCard(self, card_class):
        return any(isinstance(c, card_class) for c in self.getAllCards())
