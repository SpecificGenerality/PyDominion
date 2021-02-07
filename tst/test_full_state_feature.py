import unittest
from unittest.mock import Mock

import numpy.testing as npt
from config import GameConfig
from enums import FeatureType, StartingSplit, Zone
from game import Game
from supply import Supply


class TestFullStateFeature(unittest.TestCase):
    def setUp(self) -> None:
        self.config = GameConfig(sandbox=False, split=StartingSplit.Starting25Split, feature_type=FeatureType.FullFeature)
        self.supply = Supply(self.config)
        self.players = [Mock(), Mock()]
        self.game = Game(self.config, self.players)

    def test_full_state_feature_init(self):
        self.game.new_game()
        feature = self.game.state.feature.to_numpy()
        p0_idx = self.game.state.feature.get_player_idx(0)
        p1_idx = self.game.state.feature.get_player_idx(1)
        width = self.game.state.feature.player_width
        npt.assert_array_equal(feature[p0_idx:p0_idx + width], feature[p1_idx:p1_idx + width])
        npt.assert_equal(sum(feature[:p0_idx]), 250)

    def test_reduced_state_feature_getter(self):
        self.game.new_game()
        feature = self.game.state.feature
        # Opponent state is symmetric, so test above will fail if getters are wrong for opp
        self.assertEqual(feature.get_action_card_count(0, Zone.Hand), 0)
        self.assertEqual(feature.get_action_card_count(0, Zone.Deck), 0)
        self.assertEqual(feature.get_action_card_count(0, Zone.Play), 0)
        self.assertEqual(feature.get_action_card_count(0, Zone.Discard), 0)

        self.assertEqual(feature.get_treasure_card_count(0, Zone.Hand), 2)
        self.assertEqual(feature.get_treasure_card_count(0, Zone.Deck), 5)
        self.assertEqual(feature.get_treasure_card_count(0, Zone.Play), 0)
        self.assertEqual(feature.get_treasure_card_count(0, Zone.Discard), 0)

        self.assertEqual(feature.get_victory_card_count(0, Zone.Hand), 3)
        self.assertEqual(feature.get_victory_card_count(0, Zone.Deck), 0)
        self.assertEqual(feature.get_victory_card_count(0, Zone.Play), 0)
        self.assertEqual(feature.get_victory_card_count(0, Zone.Discard), 0)

        self.assertEqual(feature.get_total_coin_count(0, Zone.Hand), 2)
        self.assertEqual(feature.get_total_coin_count(0, Zone.Deck), 5)
        self.assertEqual(feature.get_total_coin_count(0, Zone.Play), 0)
        self.assertEqual(feature.get_total_coin_count(0, Zone.Discard), 0)

        counts = feature.get_card_counts(0)
        for card_name, count in counts.items():
            if card_name == 'Copper':
                self.assertEqual(count, 7)
            elif card_name == 'Estate':
                self.assertEqual(count, 3)
            else:
                self.assertEqual(count, 0)
