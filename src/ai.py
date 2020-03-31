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
from aiutils import *

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
                    assert next_node.parent == self.player.node
                    self.player.node = next_node
                    self.player.node.n += 1
                elif not self.expanded:
                # expand one node
                    for c in d.cardChoices + [None]:
                        if isinstance(c, Curse):
                            continue
                        leaf = Node(self.player.node)
                        leaf.card = c
                        self.player.node.children.append(leaf)
                        if c == response.singleCard:
                            next_node = leaf
                    self.expanded = True
                    self.player.node = next_node
                    self.player.node.n += 1

            s.processDecision(response)
            s.advanceNextDecision()

        # backpropagate
        player_turns = s.playerStates[0].turns
        score = -1 if player_turns >= self.T else 1 + player_turns / 10
        self.player.node.v += score
        self.player.node = self.player.node.parent
        while self.player.node != self.player.root:
            nodes = [n.v for n in self.player.node.children if n.n > 0]
            self.player.node.v = sum(nodes) / len(nodes)
            self.player.node = self.player.node.parent

        return score

    def reset(self):
        config = GameConfig(StartingSplit.StartingRandomSplit, prosperity=False, numPlayers=1)
        data = GameData(config)
        self.expanded = False

        self.game = Game(config, data, [self.player])
        self.game.newGame()

        self.player.reset(self.game.state.playerStates[0])

    def train(self, n: int, m: int, output_iters: int):
        scores = []
        for i in tqdm(range(n)):
            # initialize new game
            self.reset()
            score = self.run()
            scores.append(score)

            if i % m == 0:
                save(self.player.root, m)

            if i > 0 and i % output_iters == 0:
                print(f'Last {output_iters} avg: {sum(scores[i-output_iters:i]) / output_iters}')
                print(f'Total {i} avg: {sum(scores) / i}')

        save(self.player.root, n)

if __name__ == '__main__':
    mcts = MCTS(60)
    mcts.train(10000, 1000, 100)
