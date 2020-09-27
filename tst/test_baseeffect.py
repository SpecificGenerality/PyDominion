import unittest
from unittest.mock import Mock

from cardeffectbase import BanditEffect
from config import GameConfig
from enums import StartingSplit
from game import Game
from playerstate import PlayerState
from treasurecard import Copper, Gold, Silver
from victorycard import Estate


class TestBaseEffect(unittest.TestCase):
    def setUp(self) -> None:
        self.config = GameConfig(StartingSplit.StartingRandomSplit, False, 2)
        self.players = [Mock(), Mock()]
        self.game = Game(self.config, self.players)

    def test_bandit_effect(self) -> None:
        p_state: PlayerState = self.game.state.player_states[1]
        deck = p_state._deck
        top = Silver()
        second = Gold()
        deck[-1] = top
        deck[-2] = second

        effect = BanditEffect()

        effect.play_action(self.game.state)
        self.game.state.advance_next_decision()
        self.assertEquals(self.game.state.trash, [top])
        self.assertEquals(p_state._discard, [second])

    def test_bandit_attack_discard_only(self) -> None:
        p_state: PlayerState = self.game.state.player_states[1]
        deck = p_state._deck
        not_treasure = Estate()
        not_valuable = Copper()
        deck[-1] = not_treasure
        deck[-2] = not_valuable

        effect = BanditEffect()

        effect.play_action(self.game.state)
        self.game.state.advance_next_decision()
        self.assertEquals(self.game.state.trash, [])
        self.assertEquals(p_state._discard, [not_treasure, not_valuable])
