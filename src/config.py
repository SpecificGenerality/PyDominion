from typing import List

from card import Card
from constants import BASE_CARDS
from enums import AIConstants, GameConstants, StartingSplit


class GameConfig:
    def __init__(self, split: StartingSplit,
                 prosperity: bool,
                 num_players: int,
                 sandbox=False,
                 must_include: List[Card] = []):
        self.starting_split = split
        self.prosperity = prosperity
        self.num_players = num_players
        self.must_include = must_include
        self.sandbox = sandbox
        self.randomizers = BASE_CARDS
        self.kingdom_size = GameConstants.BaseKingdomSize

        if self.sandbox:
            self.num_cards = GameConstants.BaseSupplySize
        elif self.prosperity:
            self.num_cards = GameConstants.BaseSupplySize + self.kingdom_size + 2
        else:
            self.num_cards = GameConstants.BaseSupplySize + self.kingdom_size

        # + self.num_cards for supply
        self.feature_size = AIConstants.NumZones * self.num_cards * self.num_players + self.num_cards
