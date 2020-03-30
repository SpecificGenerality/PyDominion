from player import MCTSPlayer
from config import GameConfig
from mcts import *
from gamedata import GameData
from game import Game
from state import *
import logging
from enums import *
from tqdm import tqdm
import pickle

class MCTS:
    def __init__(self, T: int):
        self.player = MCTSPlayer()
        self.game = None
        # max number of turns in a game
        self.T = T
        self.expanded = False

    def run(self):
        s = self.game.state
        d = s.decision
        # run the game up to game end or turn limit reached
        while d.type != DecisionType.DecisionGameOver and s.playerStates[0].turns < self.T:
            if d.text:
                logging.info(d.text)
            response = DecisionResponse([])
            player = self.game.players[d.controllingPlayer]
            next_node = player.controller.makeDecision(s, response)

            if s.phase == Phase.BuyPhase:
                # apply selection until leaf node is reached
                if next_node:
                    next_node.n += 1
                    next_node.np += 1
                    self.player.node = next_node
                elif not self.expanded:
                # expand one node
                    next_node = Node(self.player.node)
                    next_node.card = response.singleCard
                    next_node.n += 1
                    next_node.np += 1
                    self.expanded = True
                    self.player.node.children.append(next_node)
                    self.player.node = next_node

            s.processDecision(response)
            s.advanceNextDecision()

        # backpropagate
        score = self.game.getPlayerScores()[0]
        while self.player.node != self.player.root:
            self.player.node.v += score
            self.player.node = self.player.node.parent

    def reset(self):
        config = GameConfig(StartingSplit.StartingRandomSplit, prosperity=False, numPlayers=1)
        data = GameData(config)
        self.expanded = False

        self.game = Game(config, data, [self.player])
        self.game.newGame()

        self.player.reset(self.game.state.playerStates[0])

    def save(self, i: int):
        with open (f'mcts_chkpt_{i}.pk1', 'wb') as output:
            pickle.dump(self.player.root, output, pickle.HIGHEST_PROTOCOL)

    def train(self, n: int, m: int):
        for i in tqdm(range(n)):
            # initialize new game
            self.reset()
            self.run()

            if i % m == 0:
                self.save(m)

        self.save(n)

if __name__ == '__main__':
    mcts = MCTS(20)
    mcts.train(10000, 1000)
