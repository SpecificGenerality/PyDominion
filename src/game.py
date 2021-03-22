import logging
from collections import Counter
from typing import List

import numpy as np

from config import GameConfig
from player import Player, PlayerInfo
from state import DecisionResponse, DecisionType, State


class Game:
    def __init__(self, config: GameConfig, players: List[Player]):
        self.config = config
        self.state = State(config)
        self.players = [PlayerInfo(i, player) for i, player in enumerate(players)]

    @property
    def done(self):
        return self.state.decision.type == DecisionType.DecisionGameOver

    def new_game(self) -> None:
        self.state = State(self.config)
        self.state.new_game()

    def get_supply_card_types(self) -> List[str]:
        return [str(c()) for c in self.state.supply.keys()]

    def get_player_scores(self) -> np.array:
        scores = np.zeros(len(self.players), dtype=int)
        for i, pInfo in enumerate(self.players):
            scores[i] = self.state.get_player_score(pInfo.id)

        return scores

    def is_winner(self, player: int) -> bool:
        scores = self.get_player_scores()
        max_score = max(scores)
        counts = Counter(scores)
        turns = [p_state.turns for p_state in self.state.player_states]
        min_turns = min(turns)

        if scores[player] < max_score:
            return False
        elif counts[max_score] == 1:
            return True
        elif turns[player] == min_turns:
            return True
        return False

    def run(self, T=None):
        d = self.state.decision
        self.state.advance_next_decision()
        while d.type != DecisionType.DecisionGameOver:
            if T is not None and all(t.turns >= T for t in self.state.player_states):
                break
            if d.text:
                logging.info(d.text)
            response = DecisionResponse([])
            player = self.players[self.state.decision.controlling_player]
            player.controller.makeDecision(self.state, response)
            self.state.process_decision(response)
            self.state.advance_next_decision()
