from game import Game
from player import MCTSPlayer
from aiutils import *
from typing import Dict

class MCTSData:
    def __init__(self):
        self.scores = []
        self.decks = []
        self.masts = {}

    def append_mast(self, M2: Dict, i: int):
        for k, v in M2.items():
            if k not in self.masts:
                self.masts[k] = [(i, v[0])]
            else:
                self.masts[k].append((i, v[0]))

    def update(self, G: Game, P: MCTSPlayer, i: int):
        self.scores.append(G.getPlayerScores()[0])
        self.decks.append(get_card_counts(G.getAllCards(0)))
        self.append_mast(P.mast, i)