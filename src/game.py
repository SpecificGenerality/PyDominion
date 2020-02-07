from config import GameConfig
from gamedata import GameData
from player import *
from playerstate import PlayerState
from enums import StartingSplit
from state import State
from typing import List
import logging

class Game:
    def __init__(self, config: GameConfig, data: GameData, players: List[Player]):
        self.gameConfig = config
        self.state = State(config, data)
        self.players = [PlayerInfo(i, player) for i, player in enumerate(players)]

        if any(isinstance(player, HumanPlayer) for player in players):
            logging.basicConfig(format='%(message)s', level=logging.INFO)
        else:
            logging.basicConfig(level=logging.WARNING)

    def newGame(self):
        self.state.newGame()

    def getWinningPlayers(self):
        scores = [self.state.getPlayerScore(pInfo.id) for pInfo in self.players]
        m = max(scores)
        return [i for i, j in enumerate(scores) if j == m]

    def getPlayerStats(self, player):
        score = self.state.getPlayerScore(player)
        counter = self.state.getCardCounts(player)
        return {'Score': score, 'Cards': counter}

    def getStats(self):
        stats = {}
        for pInfo in self.players:
            player = pInfo.id
            stats[player] = self.getPlayerStats(player)
            stats['EmptyPiles'] = []
            for k, v in self.state.data.supply.items():
                if v == 0:
                    stats['EmptyPiles'].append(k())
            stats['Winners'] = self.getWinningPlayers()
        return stats

    def run(self):
        d = self.state.decision
        while d.type != DecisionType.DecisionGameOver:
            if d.text:
                logging.info(d.text)
            response = DecisionResponse([])
            player = self.players[self.state.decision.controllingPlayer]
            player.controller.makeDecision(self.state, response)
            self.state.processDecision(response)
            self.state.advanceNextDecision()

        for pInfo in self.players:
            player = pInfo.id
            score, counter = self.getPlayerStats(player)
            logging.info(f'====Player {player} Stats====\nScore: {score}\nCards: {counter}')



