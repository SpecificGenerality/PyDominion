import unittest
from mcts import GameTree
from config import GameConfig
from enums import StartingSplit
import os
from aiconfig import model_dir
from constants import DEFAULT_KINGDOM
from player import load_players


class TestMCTSPlayer(unittest.TestCase):
    def setUp(self):
        tree_name = 'r-r-mcts-mlog-10k-score-base-default'
        rollout_name = 'mlog-10k-base-default'
        tree_path = os.path.join(model_dir, tree_name)
        rollout_path = os.path.join(model_dir, rollout_name)

        tree = GameTree.load(tree_path, train=False)

        config = GameConfig(split=StartingSplit.StartingRandomSplit, sandbox=False, num_players=2, must_include=DEFAULT_KINGDOM)
        players = load_players(['BM', 'UCT'], [rollout_path], tree=tree, train=False, rollout_type='mlog', use_tree=True)
