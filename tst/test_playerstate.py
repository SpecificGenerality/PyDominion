import unittest
from unittest.mock import Mock

from enums import StartingSplit
from playerstate import PlayerState


class TestPlayerState(unittest.TestCase):
    def setUp(self) -> None:
        config = Mock()
        config.starting_splits = [StartingSplit.StartingRandomSplit, StartingSplit.StartingRandomSplit]

        self.p_state = PlayerState(config, pid=0)

    def test_accessors(self):
        self.assertEquals(self.p_state.num_cards, 10)
