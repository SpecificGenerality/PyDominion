import random
from collections import Counter
from typing import List

import utils
from actioncard import ActionCard, Chapel
from card import Card
from config import GameConfig
from enums import StartingSplit, Zone
from treasurecard import Copper, TreasureCard
from victorycard import Estate, VictoryCard


class PlayerState:
    def __init__(self, game_config: GameConfig) -> None:
        self._actions = 1
        self._buys = 1
        self._coins = 0
        self._turns = 0
        self._deck = []
        self._discard = []
        self.hand = []
        self._island = []
        self._play_area = []

        if (game_config.starting_split == StartingSplit.Starting34Split):
        	self._deck = [Copper() for i in range(3)] + [Estate() for i in range(2)] + [Copper() for i in range(4)] + [Estate()]
        elif (game_config.starting_split == StartingSplit.Starting25Split):
        	self._deck = [Copper() for i in range(5)] + [Estate() for i in range(3)] + [Copper() for i in range(2)]
        else:
        	self._deck = [Copper() for i in range(7)] + [Estate() for i in range(3)]
        	random.shuffle(self._deck)

    @property
    def actions(self):
        return self._actions

    @actions.setter
    def actions(self, val: int):
        self._actions = val

    @property
    def buys(self):
        return self._buys

    @buys.setter
    def buys(self, val: int):
        self._buys = val

    @property
    def coins(self):
        return self._coins

    @coins.setter
    def coins(self, val: int):
        self._coins = val

    @property
    def turns(self):
        return self._turns

    @turns.setter
    def turns(self, val: int):
        self._turns = val

    @property
    def cards(self):
        cards = self.hand.copy()
        cards[0:0] = self._deck
        cards[0:0] = self._discard
        cards[0:0] = self._play_area
        cards[0:0] = self._island
        return cards

    @property
    def num_cards(self):
        return len(self.hand) + len(self._deck) + len(self._discard) + len(self._play_area) + len(self._island)

    def zone_size(self, zone: Zone):
        return len(self._get_zone_cards(zone))

    def shuffle(self) -> None:
        random.shuffle(self._discard)
        self._deck = self._deck + self._discard
        self._discard = []

    def is_degenerate(self) -> None:
        return self.num_cards == 1 and self.has_card(Chapel)

    def get_card_counts(self) -> Counter:
        cards = self.cards
        return Counter([str(card) for card in cards])

    def get_terminal_action_density(self) -> float:
        cards = self.cards
        return sum(1 if isinstance(card, ActionCard) and card.get_plus_actions() == 0 else 0 for card in cards) / len(cards)

    def get_terminal_draw_density(self) -> float:
        cards = self.cards
        return sum(1 if isinstance(card, ActionCard) and card.get_plus_actions() == 0 and card.get_plus_cards() > 0 else 0 for card in cards) / len(cards)

    def get_total_treasure_value(self) -> int:
        return sum(c.get_treasure() for c in self.cards)

    def _get_zone_cards(self, zone: Zone):
        if zone == Zone.Hand:
            return self.hand
        elif zone == Zone.Deck:
            return self._deck
        elif zone == Zone.Discard:
            return self._discard
        elif zone == Zone.Island:
            return self._island
        elif zone == Zone.Play:
            return self._play_area
        else:
            raise ValueError(f'Playerstate does not have list corresponding to zone: {zone}.')

    def get_action_card_count(self, zone: Zone) -> int:
        cards = self._get_zone_cards(zone)
        return sum(isinstance(card, ActionCard) for card in cards)

    def get_treasure_card_count(self, zone: Zone) -> int:
        cards = self._get_zone_cards(zone)
        return sum(isinstance(card, TreasureCard) for card in cards)

    def get_victory_card_count(self, zone: Zone) -> int:
        cards = self._get_zone_cards(zone)
        return sum(isinstance(card, VictoryCard) for card in cards)

    def get_total_coin_count(self, zone: Zone) -> int:
        cards = self._get_zone_cards(zone)
        return sum(card.get_plus_coins() for card in cards)

    def contains_card(self, card: Card, zone: Zone) -> bool:
        cards = self._get_zone_cards(zone)
        return utils.contains_card(card, cards)

    def has_card(self, card_class):
        return any(isinstance(c, card_class) for c in self.cards)
