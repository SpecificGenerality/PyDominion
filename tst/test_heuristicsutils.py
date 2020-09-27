import unittest

from actioncard import Festival, Merchant, Smithy
from heuristicsutils import (get_best_TD_card, get_first_no_action_card,
                             get_first_plus_action_card, get_highest_VP_card,
                             get_lowest_treasure_card, has_excess_actions,
                             has_plus_action_cards, has_treasure_cards)
from treasurecard import Gold, Silver
from victorycard import Estate


class TestHeuristicsUtils(unittest.TestCase):
    def setUp(self) -> None:
        self.choices = [Estate(), Silver(), Gold(), Merchant(), Festival(), Smithy()]

    def test_utils(self) -> None:
        self.assertEquals(get_best_TD_card(self.choices), self.choices[-1])
        self.assertTrue(has_plus_action_cards(self.choices))
        self.assertEquals(get_first_plus_action_card(self.choices), self.choices[3])
        self.assertEquals(get_lowest_treasure_card(self.choices), self.choices[1])
        self.assertTrue(has_treasure_cards(self.choices))
        self.assertEquals(get_first_no_action_card(self.choices), self.choices[-1])
        self.assertEquals(get_highest_VP_card(self.choices), self.choices[0])
        self.assertFalse(has_excess_actions(self.choices))
