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

    def testRemoveCard(self):
        card = Colony()

        cards = []

        self.assertIsNone(removeCard(card, cards))
        
        cards = [card]

        self.assertEqual(removeCard(card, cards), card)

        cards = [Estate(), card, Province(), Colony()]

        self.assertEqual(removeCard(card, cards), card)

    def testGetFirstIndex(self):
        card = Colony()

        cards = []
        self.assertEqual(getFirstIndex(card, cards), -1)

        cards = [Estate(), Province()]

        self.assertEqual(getFirstIndex(card, cards), -1)

        cards = [Colony(), Estate(), Province()]
         
        self.assertEqual(getFirstIndex(card, cards), 0)

        cards = [Estate(), Colony(), Province(), Colony()]

        self.assertEqual(getFirstIndex(card, cards), 1) 

    def test_running_mean(self):
        x, N = [1,1,1], 1
        expected = np.array([1,1,1])

        np.testing.assert_allclose(running_mean(x, N), expected)

        x, N = [1,1,1], 4
        expected = np.array([1,1,1])

        np.testing.assert_allclose(running_mean(x, N), expected)

        x, N = [1,1,1], 0
        expected = np.array([1,1,1])

        np.testing.assert_allclose(running_mean(x, N), expected)

        x, N = [1, -1, 1, -1, 1, -1], 2
        expected = np.array([1, 0, 0, 0, 0, 0])

        np.testing.assert_allclose(running_mean(x, N), expected)

        x, N = [1, -1, 1, -1, 1, -1], 3
        expected = np.array([1, 0, 1/3, -1/3, 1/3, -1/3])

        np.testing.assert_allclose(running_mean(x, N), expected)
