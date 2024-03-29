from typing import Counter, List

import pandas as pd

from card import Card
from game import Game
from victorycard import Colony, Province


class SimulationData:
    def __init__(self, cards: List[str]):
        self.player_data = []
        self.game_data = []
        self.turn_data = []
        self.action_data = []
        self.card_data = []
        self.player_df = None
        self.game_df = None
        self.turn_df = None
        self.action_df = None
        self.card_df = None
        self.cards = cards
        self.summary = {}

    def update_card(self, n: int, player: int, turn: int, cards: Counter[str]) -> None:
        card_stats = dict()
        card_stats['Iter'] = n
        card_stats['Player'] = player
        card_stats['Turn'] = turn
        for card in self.cards:
            card_stats[card] = cards.get(card, 0)
        self.card_data.append(card_stats)

    def update_turn(self, n: int, player: int, turn: int, score: int, card: Card, money_density: float) -> None:
        turn_stats = {'Iter': n, 'Player': player, 'Score': score, 'Card': str(card), 'Turn': turn, 'Density': money_density}
        self.turn_data.append(turn_stats)

    def update_action(self, n: int, player: int, turn: int, card: Card) -> None:
        action_stats = {'Iter': n, 'Player': player, 'Card': str(card), 'Turn': turn}
        self.action_data.append(action_stats)

    def update(self, G: Game, time):
        scores = G.get_player_scores()
        game_stats = dict()
        game_stats['MaxScore'] = max(scores)
        game_stats['ProvinceWin'] = G.state.supply[Province] == 0
        game_stats['TimeElapsed'] = time
        game_stats['Tie'] = G.is_winner(0) and G.is_winner(1)
        game_stats['Turns'] = G.state.player_states[0].turns

        if Colony in G.state.supply:
            game_stats['ColonyWin'] = G.state.supply[Colony] == 0

        self.game_data.append(game_stats)

        for i in range(G.config.num_players):
            sim_stats = dict()
            sim_stats['Turns'] = G.state.player_states[i].turns
            sim_stats['Player'] = i
            sim_stats['Score'] = scores[i]
            sim_stats['Won'] = G.is_winner(i)
            sim_stats['Iter'] = len(self.game_data) - 1
            self.player_data.append(sim_stats)

    def finalize(self, G: Game):
        self.player_df = pd.DataFrame(self.player_data)
        self.game_df = pd.DataFrame(self.game_data)
        self.turn_df = pd.DataFrame(self.turn_data)
        self.action_df = pd.DataFrame(self.action_data)
        self.card_df = pd.DataFrame(self.card_data)

        for i in range(G.config.num_players):
            self.summary[i] = self.player_df[self.player_df['Player'] == i]['Won'].sum()

        self.summary['ProvinceWins'] = self.game_df['ProvinceWin'].sum()
        self.summary['Ties'] = self.game_df['Tie'].sum()
