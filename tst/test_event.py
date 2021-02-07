import unittest
from unittest.mock import Mock

from actioncard import Militia, Moat, Sentry
from config import GameConfig
from enums import StartingSplit
from game import Game
from playerstate import PlayerState
from state import DecisionResponse, ReorderCards


class TestEvent(unittest.TestCase):
    def setUp(self) -> None:
        self.config = GameConfig(StartingSplit.StartingRandomSplit, prosperity=False, num_players=2, sandbox=False, must_include=[Sentry, Moat, Militia])
        self.players = [Mock(), Mock()]
        self.game = Game(self.config, self.players)

    def test_moat_reveal(self) -> None:
        self.game.new_game()

        # Inject necessary cards into players' hands
        attack_card = Militia()
        moat_card = Moat()
        self.game.state.inject(0, attack_card)
        self.game.state.inject(1, moat_card)

        self.game.state.advance_next_decision()

        # Action Phase decision
        r = DecisionResponse([])
        r.cards = [attack_card]
        self.game.state.process_decision(r)
        self.game.state.advance_next_decision()

        # MoatReveal reaction
        r = DecisionResponse([])
        r.choice = 0
        self.game.state.process_decision(r)
        self.game.state.advance_next_decision()

        self.assertEqual(self.game.state.events, [])

    def test_event_sentry(self) -> None:
        self.game.new_game()

        # Inject Sentry in player's hand
        sentry = Sentry()

        self.game.state.inject(0, sentry)

        self.game.state.advance_next_decision()

        # Action Phase Decision
        r = DecisionResponse([])
        r.cards = [sentry]
        self.game.state.process_decision(r)
        self.game.state.advance_next_decision()

        # Choose to trash one card
        d = self.game.state.decision
        trashed = d.card_choices[0]
        r = DecisionResponse([trashed])
        self.game.state.process_decision(r)
        # Trash card
        self.game.state.advance_next_decision()

        self.assertEqual(self.game.state.trash, [trashed])

        # Choose to discard one card
        d = self.game.state.decision
        discarded = d.card_choices[0]
        r = DecisionResponse([discarded])
        self.game.state.process_decision(r)
        # Discard card
        self.game.state.advance_next_decision()

        d = self.game.state.decision
        p_state: PlayerState = self.game.state.player_states[0]
        self.assertEqual(p_state._discard, [discarded])
        self.assertIsNone(d.active_card)

    def test_event_reorder_cards(self) -> None:
        p_state: PlayerState = self.game.state.player_states[0]
        deck = p_state._deck
        first = deck[-2]
        second = deck[-1]

        event = ReorderCards([], player=0)

        self.assertEqual(deck[-2], first)
        self.assertEqual(deck[-1], second)

        event = ReorderCards([second, first], player=0)

        event.advance(self.game.state)

        self.assertEqual(deck[-1], first)
        self.assertEqual(deck[-2], second)
