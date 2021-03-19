import pandas as pd

from card import Card
from game import Game
from victorycard import Colony, Province


class SimulationData:
    def __init__(self):
        self.player_data = []
        self.game_data = []
        self.turn_data = []
        self.player_df = None
        self.game_df = None
        self.turn_df = None
        self.summary = {}

    def update_turn(self, n: int, player: int, turn: int, score: int, card: Card, money_density: float):
        turn_stats = {'Iter': n, 'Player': player, 'Score': score, 'Card': str(card), 'Turn': turn, 'Density': money_density}
        self.turn_data.append(turn_stats)

    def update(self, G: Game, time):
        scores = G.get_player_scores()
        winning_score = max(scores)
        game_stats = dict()
        game_stats['MaxScore'] = winning_score
        game_stats['ProvinceWin'] = G.state.supply[Province] == 0
        game_stats['TimeElapsed'] = time
        game_stats['Tie'] = sum(1 if s == winning_score else 0 for s in scores) > 1
        game_stats['Turns'] = G.state.player_states[0].turns

        if Colony in G.state.supply:
            game_stats['ColonyWin'] = G.state.supply[Colony] == 0

        self.game_data.append(game_stats)

        for i in range(G.config.num_players):
            sim_stats = dict()
            sim_stats['Turns'] = G.state.player_states[i].turns
            sim_stats['Player'] = i
            sim_stats['Score'] = scores[i]
            sim_stats['Won'] = scores[i] >= winning_score
            sim_stats['Iter'] = len(self.game_data) - 1
            self.player_data.append(sim_stats)

    def finalize(self, G: Game):
        self.player_df = pd.DataFrame(self.player_data)
        self.game_df = pd.DataFrame(self.game_data)
        self.turn_df = pd.DataFrame(self.turn_data)

        for i in range(G.config.num_players):
            self.summary[i] = self.player_df.loc[self.player_df['Player'] == i]['Won'].sum()

        self.summary['ProvinceWins'] = self.game_df['ProvinceWin'].sum()
        self.summary['Ties'] = self.game_df['Tie'].sum()
