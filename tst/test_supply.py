import unittest

from config import GameConfig
from supply import Supply
from treasurecard import Gold
from victorycard import Province


class TestSupply(unittest.TestCase):
    def setUp(self) -> None:
        self.config = GameConfig(num_players=2, sandbox=False)
        self.supply = Supply(self.config)

    def test_supply_init(self):
        self.assertEqual(self.supply[Province], 8)
        self.assertEqual(self.supply[Gold], 30)
        self.assertEqual(len(self.supply), 17)

    def test_supply_operations(self):
        self.supply[Province] -= 1
        self.assertEqual(self.supply[Province], 7)

    def test_supply_is_game_over(self):
        self.assertFalse(self.supply.is_game_over())

        self.supply[Province] = 0
        self.assertTrue(self.supply.is_game_over())
