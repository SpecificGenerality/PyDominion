import pandas as pd

from game import *


class SimulationData:
    def __init__(self):
        self.player_data = []
        self.game_data = []
        self.player_df = None
        self.game_df = None

    def update(self, G: Game, time):
        scores = G.getPlayerScores()
        winning_score = max(scores)
        game_stats = dict()
        game_stats['MaxScore'] = winning_score
        game_stats['ProvinceWin'] = G.state.data.supply[Province] == 0
        game_stats['TimeElapsed'] = time

        if Colony in G.state.data.supply:
            game_stats['ColonyWin'] = G.state.data.supply[Colony] == 0

        self.game_data.append(game_stats)

        for i in range(G.gameConfig.numPlayers):
            sim_stats = dict()
            sim_stats['Turns'] = G.state.playerStates[i].turns
            sim_stats['Player'] = i
            sim_stats['Score'] = scores[i]
            sim_stats['Won'] = scores[i] >= winning_score
            sim_stats['Iter'] = len(self.game_data)
            self.player_data.append(sim_stats)



    def finalize(self):
        self.player_df = pd.DataFrame(self.player_data)
        self.game_df = pd.DataFrame(self.game_data)
