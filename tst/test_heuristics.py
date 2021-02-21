import unittest
from typing import List

from actioncard import Chapel
from config import GameConfig
from enums import StartingSplit, Zone
from game import Game
from player import Player, init_players
from state import DecisionResponse, State
from treasurecard import Copper
from victorycard import Estate
from actioncard import Laboratory, Village, Smithy


class TestHeuristics(unittest.TestCase):
    def setUp(self) -> None:
        self.config = GameConfig(sandbox=False, splits=[StartingSplit.Starting34Split, StartingSplit.Starting34Split])
        self.players: List[Player] = init_players(['tdebm', 'tdebm'])
        self.game = Game(self.config, self.players)

    def testChapelHeuristic(self) -> None:
        self.game.new_game()
        state: State = self.game.state

        state.inject(0, Chapel())
        state.advance_next_decision()

        # Action Phase decision: defaults to playing Chapel
        r: DecisionResponse = DecisionResponse([])
        self.players[0].makeDecision(state, r)
        self.game.state.process_decision(r)

        self.game.state.advance_next_decision()

        # Should auto trash 3 Copper and 1 Estate
        r = DecisionResponse([])
        self.players[0].makeDecision(state, r)
        self.game.state.process_decision(r)

        # Process TrashCard events
        self.game.state.advance_next_decision()

        self.assertEqual(state.get_card_count(0, Copper), 4)
        self.assertEqual(state.get_card_count(0, Estate), 2)

    def testGreedyActionHeuristic(self) -> None:
        self.game.new_game()
        state: State = self.game.state

        state.inject(0, Laboratory())
        state.inject(0, Village())
        state.inject(0, Smithy())

        state.advance_next_decision()

        # Action Phase: Play Lab
        r = DecisionResponse([])
        self.players[0].makeDecision(state, r)
        self.assertTrue(isinstance(r.cards[0], Laboratory))
        state.process_decision(r)

        state.advance_next_decision()

        r = DecisionResponse([])
        self.players[0].makeDecision(state, r)
        self.assertTrue(isinstance(r.cards[0], Village))
        state.process_decision(r)

        state.advance_next_decision()

        r = DecisionResponse([])
        self.players[0].makeDecision(state, r)
        self.assertTrue(isinstance(r.cards[0], Smithy))
        state.process_decision(r)

        state.advance_next_decision()

        self.assertEqual(state.get_zone_card_count(0, Zone.Hand), 10)
