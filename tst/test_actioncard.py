import unittest
from unittest.mock import Mock

from actioncard import Merchant
from config import GameConfig
from enums import StartingSplit
from game import Game
from playerstate import PlayerState
from state import DecisionResponse
from treasurecard import Silver
from victorycard import Estate


class TestActionCard(unittest.TestCase):
    def setUp(self) -> None:
        self.config = GameConfig(StartingSplit.StartingRandomSplit, False, 2)
        self.players = [Mock(), Mock()]
        self.game = Game(self.config, self.players)

    def test_merchant(self) -> None:
        self.game.new_game()

        p_state: PlayerState = self.game.state.player_states[0]

        # Inject cards into hand
        merchant = Merchant()
        first_silver = Silver()
        second_silver = Silver()
        p_state.hand[0] = merchant
        p_state.hand[1] = first_silver
        p_state.hand[2] = second_silver
        p_state.hand[3] = Estate()
        p_state.hand[4] = Estate()

        self.game.state.advance_next_decision()

        # Action Phase Decision -- Play Merchant
        r = DecisionResponse([merchant])
        self.game.state.process_decision(r)
        self.game.state.advance_next_decision()

        # Treasure Phase Decision -- Play All Treasures
        r = DecisionResponse([first_silver])
        self.game.state.process_decision(r)
        self.game.state.advance_next_decision()

        r = DecisionResponse([second_silver])
        self.game.state.process_decision(r)
        self.game.state.advance_next_decision()

        self.assertEqual(p_state.coins, 5)
