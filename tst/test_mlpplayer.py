import unittest
from unittest.mock import Mock

import torch
from config import GameConfig
from constants import SANDBOX_CARDS
from enums import StartingSplit
from game import Game
from mlp import SandboxMLP
from player import MLPPlayer
from treasurecard import Copper
from victorycard import Estate


class TestMLPPlayer(unittest.TestCase):
    def setUp(self) -> None:
        self.mlp = Mock()
        self.mlp.D_in = 25
        self.mlp_player = MLPPlayer(mlp=self.mlp, cards=[card_class() for card_class in SANDBOX_CARDS], n_players=2)
        self.config = GameConfig(StartingSplit.Starting25Split, False, 2, sandbox=True)
        self.players = [self.mlp_player, Mock()]
        self.game = Game(self.config, self.players)

    def test_featurize(self) -> None:
        self.game.new_game()

        x = self.mlp_player.featurize(self.game.state)
        self.assertEqual(x[self.mlp_player.idxs[str(Copper())]], 7/10)
        self.assertEqual(x[self.mlp_player.idxs[str(Estate())]], 3/10)
        self.assertEqual(x[self.mlp_player.idxs[str(Copper())]+16], 7/10)
        self.assertEqual(x[self.mlp_player.idxs[str(Estate())]+16], 3/10)

        x = self.mlp_player.featurize(self.game.state, lookahead=True, lookahead_card=Copper())
        self.assertEqual(x[self.mlp_player.idxs[str(Copper())]], 8/11)
        self.assertEqual(x[self.mlp_player.idxs[str(Estate())]], 3/11)
        self.assertEqual(x[self.mlp_player.num_cards], 1)
        self.assertEqual(x[self.mlp_player.idxs[str(Copper())]+16], 7/10)
        self.assertEqual(x[self.mlp_player.idxs[str(Estate())]+16], 3/10)

        x = self.mlp_player.featurize(self.game.state, lookahead=True, lookahead_card=Estate())
        self.assertEqual(x[self.mlp_player.idxs[str(Estate())]], 4/11)
        self.assertEqual(x[self.mlp_player.num_cards+1], 4)
    
    def test_get_expected_hand(self) -> None: 
        deck = [Copper()] * 7 + [Estate()] * 3
        expected_hand = self.mlp_player.get_expected_hand(deck)
        self.assertEqual(expected_hand[self.mlp_player.idxs[str(Copper())]], 7/2)
        self.assertEqual(expected_hand[self.mlp_player.idxs[str(Estate())]], 3/2)
