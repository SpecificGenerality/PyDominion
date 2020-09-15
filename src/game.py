import logging
from typing import List

import numpy as np

from config import GameConfig
from enums import StartingSplit
from supply import Supply
from player import *
from playerstate import PlayerState
from state import State


class Game:
    def __init__(self, config: GameConfig, players: List[Player]):
        self.config = config
        self.state = State(config)
        self.players = [PlayerInfo(i, player) for i, player in enumerate(players)]

        if any(isinstance(player, HumanPlayer) for player in players):
            logging.basicConfig(format='%(message)s', level=logging.INFO)
        else:
            logging.basicConfig(level=logging.WARNING)

    def new_game(self):
        self.state = State(self.config)
        self.state.new_game()

    def get_supply_card_types(self):
        return [str(c()) for c in self.state.supply.supply.keys()]

    def get_winning_players(self):
        scores = [self.state.get_player_score(pInfo.id) for pInfo in self.players]
        m = max(scores)
        return [i for i, j in enumerate(scores) if j == m]

    def get_player_scores(self):
        scores = np.zeros(len(self.players))
        for i, pInfo in enumerate(self.players):
            scores[i] = self.state.get_player_score(pInfo.id)

        return scores

    def run(self, T=None):
        d = self.state.decision
        while d.type != DecisionType.DecisionGameOver:
            if T is not None and all(t.turns >= T for t in self.state.player_states):
                break
            if d.text:
                logging.info(d.text)
            response = DecisionResponse()
            player = self.players[self.state.decision.controlling_player]
            player.controller.makeDecision(self.state, response)
            self.state.process_decision(response)
            self.state.advance_next_decision()
