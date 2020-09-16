import unittest
from unittest.mock import Mock

from actioncard import Militia, Moat
from config import GameConfig
from enums import StartingSplit
from game import Game
from player import Player
from state import DecisionResponse, DecisionState, MoatReveal
from supply import Supply
from treasurecard import Copper


class TestEvent(unittest.TestCase):
    def setUp(self) -> None:
        self.config = GameConfig(StartingSplit.StartingRandomSplit, False, 2)
        self.players = [Mock(), Mock()]
        self.game = Game(self.config, self.players)

    def test_moat_reveal(self) -> None:
        r = DecisionResponse()

        self.game.new_game()

        # Inject necessary cards into players' hands
        attack_card = Militia()
        moat_card = Moat()
        self.game.state.player_states[0].hand[0] = attack_card
        self.game.state.player_states[1].hand[0] = moat_card

        self.game.state.advance_next_decision()

        # Action Phase decision
        r = DecisionResponse()
        r.cards = [attack_card]
        self.game.state.process_decision(r)
        self.game.state.advance_next_decision()

        # MoatReveal reaction
        r = DecisionResponse()
        r.choice = 0
        self.game.state.process_decision(r)
        self.game.state.advance_next_decision()

        self.assertEqual(self.game.state.events, [])
