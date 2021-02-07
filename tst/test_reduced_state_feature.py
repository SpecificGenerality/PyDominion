import unittest
from unittest.mock import Mock

import numpy.testing as npt
from config import GameConfig
from enums import FeatureType, StartingSplit
from game import Game
from supply import Supply


class TestReducedStateFeature(unittest.TestCase):
    def setUp(self) -> None:
        self.config = GameConfig(sandbox=False, split=StartingSplit.Starting25Split, feature_type=FeatureType.ReducedFeature)
        self.supply = Supply(self.config)
        self.players = [Mock(), Mock()]
        self.game = Game(self.config, self.players)

    def test_reduced_state_feature_init(self):
        self.game.new_game()
        feature = self.game.state.feature.to_numpy()
        npt.assert_array_equal(feature[17:34], feature[34:])
        npt.assert_equal(sum(feature[:17]), 250)
