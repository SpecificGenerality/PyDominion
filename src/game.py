from config import GameConfig
from gamedata import GameData
from player import *
from playerstate import PlayerState
from enums import StartingSplit
from state import State

class Game:
    def __init__(self, config: GameConfig, data: GameData):
        self.gameConfig = config
        self.state = State(config, data)
        self.players = [PlayerInfo(i, HumanPlayer()) for i in range(config.numPlayers)]

    def newGame(self):
        self.state.newGame()

    def run(self):
        d = self.state.decision
        while d.type != DecisionType.DecisionGameOver:
            if d.text:
                print(d.text)
            response = DecisionResponse([])
            # print(response)
            player = self.players[self.state.player]
            print(f'{player} \n')
            player.controller.makeDecision(self.state, response)
            # print(response)
            self.state.processDecision(response)
            self.state.advanceNextDecision()
