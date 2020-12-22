from typing import List

from card import Card
from constants import BASE_CARDS
from enums import AIConstants, GameConstants, StartingSplit, FeatureType


class GameConfig:
    def __init__(self, split: StartingSplit,
                 prosperity: bool,
                 num_players: int,
                 sandbox=False,
                 must_include: List[Card] = [],
                 feature_type: FeatureType = FeatureType.FullFeature,
                 device: str = 'cpu'):
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

        self.feature_type = feature_type
        self.device = device
        # + self.num_cards for supply
        if feature_type == FeatureType.FullFeature:
            self.feature_size = AIConstants.NumZones * self.num_cards * self.num_players + self.num_cards
        elif feature_type == FeatureType.ReducedFeature:
            self.feature_size = (self.num_players + 1) * self.num_cards
