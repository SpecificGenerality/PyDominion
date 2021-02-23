import os
import unittest

from config import GameConfig
from constants import DEFAULT_KINGDOM
from env import DefaultEnvironment
from mcts import GameTree
from mctsrunner import train_mcts
from player import init_players


class TestMCTSRunner(unittest.TestCase):
    def setUp(self) -> None:
        self.config = GameConfig(sandbox=False, num_players=2, must_include=DEFAULT_KINGDOM)
        self.tree = GameTree(train=True)
        self.player = init_players(['UCT'], tree=self.tree, rollout_type='mlog')[0]
        self.players = [self.player, self.player]

        self.project_root = '/home/justiny/Documents/Projects/PyDominion'
        self.model_dir = os.path.join(self.project_root, 'models')
        self.env = DefaultEnvironment(self.config, self.players)
        self.tree_name = 'mcts-test'
        self.rollout_name = 'mlog-test'
        self.tree_path = os.path.join(self.model_dir, self.tree_name)
        self.rollout_path = os.path.join(self.model_dir, self.rollout_name)

    def test_rollout_train(self) -> None:
        train_mcts(self.env, tree=self.tree, path=self.tree_path, rollout_path=self.rollout_path, epochs=10, train_epochs_interval=11)
