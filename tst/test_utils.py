import unittest
import numpy as np
from utils import contains_card, remove_card, remove_first_card, move_card, running_mean, get_card
from victorycard import Estate, Province, Colony


class TestUtils(unittest.TestCase):
    def test_get_card(self):
        card = Colony()
        card_type = Colony

        cards = []
        self.assertEqual(get_card(card, cards), None)
        self.assertEqual(get_card(card_type, cards), None)

        cards = [Province(), Colony()]
        self.assertEqual(get_card(card, cards), cards[-1])
        self.assertEqual(get_card(card_type, cards), cards[-1])
        self.assertEqual(get_card(Estate, cards), None)

    def test_contains_card(self):
        card = Colony()
        cards = []

        self.assertFalse(contains_card(card, cards))

        cards = [Colony() for i in range(2)]

        self.assertTrue(contains_card(card, cards))

    def test_remove_first_card(self):
        card = Colony()

        cards = []

        self.assertIsNone(remove_first_card(card, cards))

        cards = [card]

        self.assertEqual(remove_first_card(card, cards), card)

        cards = [Estate(), card, Province(), Colony()]

        self.assertEqual(remove_first_card(card, cards), card)

    def test_remove_card(self):
        card = Colony()

        cards = []

        self.assertIsNone(remove_card(card, cards), card)

        cards = [card]

        self.assertEqual(remove_card(card, cards), card)

        cards = [Colony(), Colony(), card]

        self.assertEqual(remove_card(card, cards), card)

    def test_move_card(self):
        card = Colony()

        src, dst = [], []

        self.assertRaises(ValueError, move_card, card, src, dst)

        src = [Colony()]

        self.assertRaises(ValueError, move_card, card, src, dst)

        src = [card]

        move_card(card, src, dst)
        self.assertEqual(src, [])
        self.assertEqual(dst, [card])

    def test_running_mean(self):
        x, N = [1, 1, 1], 1
        expected = np.array([1, 1, 1])

        np.testing.assert_allclose(running_mean(x, N), expected)

        x, N = [1, 1, 1], 4
        expected = np.array([1, 1, 1])

        np.testing.assert_allclose(running_mean(x, N), expected)

        x, N = [1, 1, 1], 0
        expected = np.array([1, 1, 1])

        np.testing.assert_allclose(running_mean(x, N), expected)

        x, N = [1, -1, 1, -1, 1, -1], 2
        expected = np.array([1, 0, 0, 0, 0, 0])

        np.testing.assert_allclose(running_mean(x, N), expected)

        x, N = [1, -1, 1, -1, 1, -1], 3
        expected = np.array([1, 0, 1 / 3, -1 / 3, 1 / 3, -1 / 3])

        np.testing.assert_allclose(running_mean(x, N), expected)
