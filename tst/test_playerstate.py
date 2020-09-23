import unittest
from unittest.mock import Mock

from enums import StartingSplit, Zone
from playerstate import PlayerState
from treasurecard import Copper, Silver


class TestPlayerState(unittest.TestCase):
    def setUp(self) -> None:
        config = Mock()
        config.starting_split = StartingSplit.StartingRandomSplit

        self.p_state = PlayerState(config)

    def test_accessors(self):
        self.assertEquals(self.p_state.num_cards, 10)
        self.assertEquals(self.p_state.get_total_treasure_value(), 7)
        self.assertEquals(self.p_state.get_victory_card_count(Zone.Deck), 3)
        self.assertEquals(self.p_state.get_treasure_card_count(Zone.Deck), 7)
        self.assertTrue(self.p_state.has_card(Copper))
        self.assertFalse(self.p_state.has_card(Silver))
