
import pandas as pd

from game import Game
from player import MCTSPlayer
from utils import running_mean


class MCTSData:
    def __init__(self):
        self.data = []
        self.scores = []
        self.data_df = pd.DataFrame()
        self.split_scores = []
        self.split_scores_df = pd.DataFrame()

    def update(self, G: Game, P: MCTSPlayer, i: int):
        '''Update game statistics after game i'''
        self.scores.append(G.get_player_scores()[0])
        data_dict = {}
        data_dict['score'] = G.get_player_scores()[0]
        data_dict['rollout'] = str(P.rollout)
        data_dict['i'] = i

        card_counts = G.state.get_player_card_counts(0)
        supply_cards = G.get_supply_card_types()
        for k in supply_cards:
            data_dict[k] = card_counts.get(k, 0)

        P.rollout.augment_data(data_dict)
        self.data.append(data_dict)

    def update_split_scores(self, score: int, rollout: bool, i: int):
        '''Update the scores obtained during selection and rollout, resp.'''
        data_dict = {}
        data_dict['i'] = i
        data_dict['score'] = score
        data_dict['rollout'] = rollout
        self.split_scores.append(data_dict)

    def augment_avg_scores(self, N: int):
        '''Augment data dataframe with running mean of scores with window N'''
        means = running_mean(self.data_df['score'].to_numpy(), N)
        self.data_df['avg'] = pd.Series(means, index=self.data_df.index)

    def update_dataframes(self, reset_score=True):
        '''Append current data to dataframes. Useful for updating data after games.'''
        self.data_df = self.data_df.append(pd.DataFrame(self.data))
        self.data = []
        self.split_scores_df = self.split_scores_df.append(pd.DataFrame(self.split_scores))
        self.split_scores = []
        if reset_score:
            self.scores = []
