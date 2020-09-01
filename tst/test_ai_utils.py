import unittest 
import numpy as np

from aiutils import best_path, get_branching_factor_stats
from mcts import Node 

class TestAIUtils(unittest.TestCase):
    def setUp(self):
        self.root = Node(n=5)
        self.best_child = Node(parent=self.root, v=10, n=2)
        children = [self.best_child, Node(parent=self.root, v=0, n=1), Node(parent=self.root, v=-10, n=1)]
        self.root.children = children
        self.best_leaf = Node(parent=self.best_child, v=10, n=1)
        best_node_children = [self.best_leaf, Node(parent=self.best_child, v=5, n=1)]
        self.best_child.children = best_node_children

    def test_best_path(self):
        self.assertEqual(best_path(self.root), [self.root, self.best_child, self.best_leaf])

    def test_get_branching_factors(self):
        np.testing.assert_allclose(get_branching_factor_stats(self.root), (2 / 3, 4/3))
