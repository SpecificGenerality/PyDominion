import unittest 
from player import MLPPlayer
from unittest.mock import Mock 
from constants import SANDBOX_CARDS

class TestPlayer(unittest.TestCase):
    def setUp(self) -> None: 
        self.player_one = MLPPlayer(Mock(), [card_class() for card_class in SANDBOX_CARDS], 2)
        self.player_two = MLPPlayer(Mock(), [card_class() for card_class in SANDBOX_CARDS], 2) 

    def test_mlpplayer(self) -> None: 
        self.assertEqual(self.player_one.idxs, self.player_two.idxs)