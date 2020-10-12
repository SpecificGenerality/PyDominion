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
        self.mlp.D_in = 18
        self.mlp_player = MLPPlayer(mlp=self.mlp, cards=[card_class() for card_class in SANDBOX_CARDS], n_players=2)
        self.config = GameConfig(StartingSplit.StartingRandomSplit, False, 2, sandbox=True)
        self.players = [self.mlp_player, Mock()]
        self.game = Game(self.config, self.players)

    def test_featurize(self) -> None:
        self.game.new_game()

        x = self.mlp_player.featurize(self.game.state)
        tgt_x = torch.zeros(self.mlp.D_in).cuda()
        tgt_x[self.mlp_player.idxs[str(Copper())]] = 7 / 10
        tgt_x[self.mlp_player.idxs[str(Estate())]] = 3 / 10
        tgt_x[8] = 0
        tgt_x[9] = 3
        tgt_x[self.mlp_player.idxs[str(Copper())]+10] = 7 / 10
        tgt_x[self.mlp_player.idxs[str(Estate())]+10] = 3 / 10
        tgt_x[-2] = 0
        tgt_x[-1] = 3
        self.assertTrue(torch.equal(x, tgt_x))

        x = self.mlp_player.featurize(self.game.state, Copper())
        tgt_x[self.mlp_player.idxs[str(Copper())]] = 8 / 11
        tgt_x[self.mlp_player.idxs[str(Estate())]] = 3 / 11
        tgt_x[8] += 1
        self.assertTrue(torch.equal(x, tgt_x))
        tgt_x[self.mlp_player.idxs[str(Copper())]] = 7 / 11


        x = self.mlp_player.featurize(self.game.state, Estate())
        tgt_x[self.mlp_player.idxs[str(Estate())]] = 4 / 11
        tgt_x[9] += 1
        self.assertTrue(torch.equal(x, tgt_x))
