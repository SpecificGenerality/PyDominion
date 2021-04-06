import unittest

from actioncard import Chapel, Witch
from cursecard import Curse
from treasurecard import Copper, Gold, Silver
from victorycard import Duchy, Estate, Province, VictoryCard


class TestMisc(unittest.TestCase):
    def setUp(self):
        return super().setUp()

    def test_uct_filter(self):
        choices = [Copper(), Silver(), Gold(), Curse(), Estate(), Duchy(), Province(), Chapel(), Witch()]
        filtered_choices = list(filter(lambda x: not isinstance(x, Curse) and not isinstance(x, Copper) and not issubclass(type(x), VictoryCard), choices))

        self.assertEqual(len(filtered_choices), 4)
