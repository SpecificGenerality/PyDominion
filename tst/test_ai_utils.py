import unittest
import numpy as np

from aiutils import best_path, get_branching_factor_stats, get_path, get_most_visited_paths_at_depth
from mcts import Node

class TestAIUtils(unittest.TestCase):
    def setUp(self):
        self.root = Node(n=5)
        self.root.parent = self.root

        # First level of test tree
        self.best_child = Node(parent=self.root, v=10, n=2)
        children = [self.best_child, Node(parent=self.root, v=0, n=1), Node(parent=self.root, v=-10, n=1)]
        self.root.children = children

        # Second level of test tree
        self.best_grandchild = Node(parent=self.best_child, v=10, n=1)
        best_child_children = [self.best_grandchild, Node(parent=self.best_child, v=5, n=1)]
        self.best_child.children = best_child_children

        # Third (leaf) level of test tree
        self.test_leaf = Node(parent=self.best_grandchild, v=1)
        self.best_grandchild.children = [self.test_leaf, Node(parent=self.best_grandchild)]

    def test_best_path(self):
        self.assertEqual(best_path(self.root), [self.root, self.best_child, self.best_grandchild, self.test_leaf])

    def test_get_branching_factors(self):
        np.testing.assert_allclose(get_branching_factor_stats(self.root), (2 / 3, 4/3))

    def test_get_path(self):
        self.assertEqual(get_path(self.root, self.test_leaf), [self.root, self.best_child, self.best_grandchild, self.test_leaf])

    def test_get_most_visited_paths_at_depth(self):
        self.assertEqual(get_most_visited_paths_at_depth(self.root, 1, 2), \
            [[self.best_child, self.best_grandchild], [self.best_child, self.best_child.children[1]]])
