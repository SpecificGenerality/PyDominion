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
    def __init__(self, config: GameConfig, supply: Supply, players: List[Player]):
        self.gameConfig = config
        self.state = State(config, supply)
        self.players = [PlayerInfo(i, player) for i, player in enumerate(players)]

        if any(isinstance(player, HumanPlayer) for player in players):
            logging.basicConfig(format='%(message)s', level=logging.INFO)
        else:
            logging.basicConfig(level=logging.WARNING)

    def newGame(self):
        self.state.newGame()

    def getSupplyCardTypes(self):
        return [str(c()) for c in self.state.supply.supply.keys()]

    def getWinningPlayers(self):
        scores = [self.state.getPlayerScore(pInfo.id) for pInfo in self.players]
        m = max(scores)
        return [i for i, j in enumerate(scores) if j == m]

    def getAllCards(self, player):
        return self.state.playerStates[player].cards

    def getPlayerScores(self):
        scores = np.zeros(len(self.players))
        for i, pInfo in enumerate(self.players):
            scores[i] = self.state.getPlayerScore(pInfo.id)

        return scores

    def run(self, T=None):
        d = self.state.decision
        while d.type != DecisionType.DecisionGameOver:
            if T and all(t.turns >= T for t in self.state.playerStates):
                break
            if d.text:
                logging.info(d.text)
            response = DecisionResponse([])
            player = self.players[self.state.decision.controllingPlayer]
            player.controller.makeDecision(self.state, response)
            self.state.processDecision(response)
            self.state.advanceNextDecision()
