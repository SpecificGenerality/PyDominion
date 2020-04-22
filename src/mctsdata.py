from game import Game
from player import MCTSPlayer
from aiutils import *
from typing import Dict
import pandas as pd
from utils import *

class MCTSData:
    def __init__(self):
        self.data = []
        self.masts = []
        self.scores = []
        self.data_df = pd.DataFrame()
        self.masts_df = pd.DataFrame()
        self.split_scores = [[], []]

    def update(self, G: Game, P: MCTSPlayer, i: int):
        '''Update game statistics after game i'''
        self.scores.append(G.getPlayerScores()[0])
        data_dict = {}
        data_dict['score'] = G.getPlayerScores()[0]
        data_dict['tau'] = P.tau
        data_dict['i'] = i

        card_counts = get_card_counts(G.getAllCards(0))
        supply_cards = G.getSupplyCardTypes()
        for k in supply_cards:
            data_dict[k] = card_counts.get(k, 0)

        mast_dict = {}
        mast_dict['tau'] = P.tau
        mast_dict['i'] = i
        for k in supply_cards + [str(None)]:
            mast_dict[k] = P.mast.get(k, (0, 0))[0]

        self.data.append(data_dict)
        self.masts.append(mast_dict)

    def update_split_scores(self, score: int, rollout: bool):
        '''Update the scores obtained during selection and rollout, resp.'''
        if rollout:
            self.split_scores[1].append(score)
        else:
            self.split_scores[0].append(score)

    def get_last_mast(self) -> dict:
        '''Return the mast from the end of training'''
        mast = {}
        for k, v in self.masts[-1].items():
            mast[k] = (v, None)
        return mast

    def augment_avg_scores(self, N: int):
        '''Augment data dataframe with running mean of scores with window N'''
        means = running_mean(self.data_df['score'].to_numpy(), N)
        self.data_df['avg'] = pd.Series(means, index=self.data_df.index)

    def update_dataframes(self, reset_score=True):
        '''Append current data to dataframes. Useful for updating data after games.'''
        self.data_df = self.data_df.append(pd.DataFrame(self.data))
        self.data = []
        self.masts_df= self.masts_df.append(pd.DataFrame(self.masts))
        self.data = []
        if reset_score:
            self.scores = []