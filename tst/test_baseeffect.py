import unittest
from unittest.mock import Mock

from actioncard import Bandit
from cardeffectbase import BanditEffect, VassalEffect
from config import GameConfig
from game import Game
from playerstate import PlayerState
from state import DecisionResponse
from treasurecard import Copper, Gold, Silver
from victorycard import Estate


class TestBaseEffect(unittest.TestCase):
    def setUp(self) -> None:
        self.config = GameConfig(sandbox=False, must_include=[Bandit])
        self.players = [Mock(), Mock()]
        self.game = Game(self.config, self.players)

    def test_bandit_effect(self) -> None:
        self.game.new_game()
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
        self.game.new_game()
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

    def test_vassal_effect(self) -> None:
        self.game.new_game()
        p_state: PlayerState = self.game.state.player_states[0]
        top = p_state._deck[-1]
        effect = VassalEffect()

        # Test discard
        effect.play_action(self.game.state)
        self.game.state.advance_next_decision()
        self.assertEquals(p_state._discard, [top])

    def test_vassal_effect_play_action(self) -> None:
        self.game.new_game()
        p_state: PlayerState = self.game.state.player_states[0]
        opp_state: PlayerState = self.game.state.player_states[1]
        card = Bandit()
        p_state._deck[-1] = card
        first_discarded = opp_state._deck[-1]
        second_discarded = opp_state._deck[-2]
        effect = VassalEffect()

        # Play Bandit
        r = DecisionResponse([], 1)
        effect.play_action(self.game.state)
        self.game.state.advance_next_decision()
        self.game.state.process_decision(r)

        # Process Bandit events
        self.game.state.advance_next_decision()
        self.assertIn(card, p_state._play_area)
        self.assertIn(first_discarded, opp_state._discard)
        self.assertIn(second_discarded, opp_state._discard)
