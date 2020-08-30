import unittest
from utils import *
from victorycard import *

class TestUtils(unittest.TestCase):
    def testContainsCard(self): 
        card = Colony()
        cards = []

        self.assertFalse(containsCard(card, cards))

        cards = [Colony() for i in range(2)]

        self.assertTrue(containsCard(card, cards))
