import unittest

from mcts import Node
from victorycard import Duchy, Estate, Province, Colony


class TestMCTS(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Node()

    def test_is_leaf(self) -> None:
        self.assertTrue(self.root.is_leaf())

    def test_add_children(self) -> None:
        children = [Estate(), Duchy(), Province()]
        self.root.add_unique_children(children)

        self.assertEquals(self.root.children[0].parent, self.root)
        self.assertEquals(self.root.children[1].parent, self.root)
        self.assertEquals(self.root.children[2].parent, self.root)

        self.root.add_unique_children(children)
        self.assertEquals(len(self.root.children), len(children))

    def test_update_value(self) -> None:
        children = [Estate(), Duchy(), Province()]
        self.root.add_unique_children(children)
        for child in self.root.children:
            child.n = 1
            child.v = 1

        f = lambda x: sum([y+1 for y in x])
        self.root.update_v(f)

        self.assertEquals(self.root.v, 6)

    def test_get_child(self) -> None:
        children = [Estate(), Duchy(), Province()]
        self.root.add_unique_children(children)

        self.assertIsNotNone(self.root.get_child_node(Estate()))
        self.assertIsNone(self.root.get_child_node(Colony()))
