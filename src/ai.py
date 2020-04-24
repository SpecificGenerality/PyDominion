import logging
import os
import pickle
from argparse import ArgumentParser

import numpy as np
from tqdm import tqdm

from aiutils import *
from config import GameConfig
from enums import *
from game import Game
from gamedata import GameData
from mcts import *
from mctsdata import MCTSData
from player import MCTSPlayer
from state import *

# define first k turns, and then plot the expected value
# of the random rollout
data_dir = 'C:\\Users\\yanju\\Documents\\Princeton\\IW\\Dominion\\PyDominion\\data'
model_dir = 'C:\\Users\\yanju\\Documents\\Princeton\\IW\\Dominion\\PyDominion\\models'

class MCTS:
    def __init__(self, T: int, tau: float, rollout: Rollout):
        self.player = MCTSPlayer(train=True, tau=tau, rollout=rollout)
        self.game = None
        # max number of turns in a game
        self.T = T
        self.expanded = False
        self.rollout = []
        self.data = MCTSData()

    def run(self):
        s = self.game.state
        d = s.decision
        tree_score = 0
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
                    # Uncomment to track UCT score within the tree
                    # tree_score = self.game.getPlayerScores()[0]
                    # self.data.update_split_scores(tree_score, False)
                else:
                    self.rollout.append(response.singleCard)

            s.processDecision(response)
            s.advanceNextDecision()


        player_turns = s.playerStates[0].turns
        score = self.game.getPlayerScores()[0]
        # update data
        # self.data.update_split_scores(score - tree_score, True)

        # backpropagate
        delta = score
        self.player.node.v += delta
        self.player.node = self.player.node.parent
        while self.player.node != self.player.root:
            self.player.node.update_v(lambda x: sum(x)/ len(x))
            self.player.node = self.player.node.parent

        # update history heuristic
        self.player.update_mast(self.rollout, delta)

        return self.game.getPlayerScores()[0]

    def reset(self):
        config = GameConfig(StartingSplit.StartingRandomSplit, prosperity=False, numPlayers=1)
        data = GameData(config)
        self.expanded = False
        self.rollout = []

        self.game = Game(config, data, [self.player])
        self.game.newGame()

        self.player.reset(self.game.state.playerStates[0])

    def train(self, n: int, output_iters: int,
        save_model=False, model_dir=model_dir, model_name='mcts',
        save_data=False, data_dir=data_dir, data_name='data'):

        avg = 0
        last_avg = 0
        for i in tqdm(range(n)):
            # initialize new game
            self.reset()
            self.run()
            self.data.update(self.game, self.player, i)

            avg = sum(self.data.scores) / (i+1)

            if i > 0 and i % output_iters == 0:
                print(f'Last {output_iters} avg: {sum(self.data.scores[i-output_iters:i]) / output_iters}')
                print(f'Total {i} avg: {avg}')

        if save_model:
            save(os.path.join(model_dir, model_name), self.player.root)
        if save_data:
            self.data.update_dataframes()
            save(os.path.join(data_dir, model_name), self.data)


if __name__ == '__main__':
    parser = ArgumentParser('MCTS Dominion AI')
    parser.add_argument('-T', default=30, type=int, help='Threshold number of turns')
    parser.add_argument('-n', default=10000, type=int, help='Number of training iterations')
    parser.add_argument('-l', default=100, type=int, help='Number of iterations before logging')
    parser.add_argument('-tau', default=0.5, help='Tau parameter for history heuristic Gibbs distribution')
    parser.add_argument('-rollout', default=2, type=int, help='1: Random, 2: History Heuristic 3: OLS')
    parser.add_argument('--save_model', action='store_true')
    parser.add_argument('--model_dir', type=str, help='Where to save the model', default=model_dir)
    parser.add_argument('--model_name', type=str, help='What to name the model')
    parser.add_argument('--save_data', action='store_true')
    parser.add_argument('--data_dir', type=str, help='Where to save the data', default=data_dir)
    args = parser.parse_args()

    if args.rollout == 1:
        rollout = Rollout.Random
    elif args.rollout == 2:
        rollout = Rollout.HistoryHeuristic
    elif args.rollout == 3:
        rollout = Rollout.OLSRegression

    mcts = MCTS(args.T, args.tau, rollout)
    mcts.train(args.n, args.l,
        save_model=args.save_model, model_dir=model_dir, model_name=args.model_name,
        save_data=args.save_data, data_dir=data_dir)
