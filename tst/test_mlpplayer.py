import unittest
from unittest.mock import Mock

from config import GameConfig
from constants import SANDBOX_CARDS
from enums import StartingSplit
from game import Game
from player import MLPPlayer


class TestMLPPlayer(unittest.TestCase):
    def setUp(self) -> None:
        self.config = GameConfig(StartingSplit.Starting25Split, False, 2, sandbox=True)
        self.mlp = Mock()
        self.mlp.D_in = 25
        self.mlp_player = MLPPlayer(mlp=self.mlp, cards=[card_class() for card_class in SANDBOX_CARDS], n_players=2)
        self.players = [self.mlp_player, Mock()]
        self.game = Game(self.config, self.players)
