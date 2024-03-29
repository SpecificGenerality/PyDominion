from typing import List, Type

from card import Card
from constants import BASE_CARDS, DEFAULT_KINGDOM
from enums import AIConstants, GameConstants, StartingSplit, FeatureType


class GameConfig:
    def __init__(self, splits: List[StartingSplit] = [StartingSplit.StartingRandomSplit, StartingSplit.StartingRandomSplit],
                 prosperity: bool = False,
                 num_players: int = 2,
                 sandbox=True,
                 must_include: List[Type[Card]] = DEFAULT_KINGDOM,
                 feature_type: FeatureType = FeatureType.ReducedFeature,
                 device: str = 'cpu'):
        self.starting_splits: List[StartingSplit] = splits
        self.prosperity = prosperity
        self.num_players = num_players
        self.must_include = must_include if not sandbox else must_include
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
