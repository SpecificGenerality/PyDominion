from enums import GameConstants, StartingSplit
from constants import BASE_CARDS


class GameConfig:
    def __init__(self, split: StartingSplit, prosperity: bool, num_players:int, sandbox=False):
        self.randomizers = BASE_CARDS
        self.kingdom_size = GameConstants.KingdomSize
        self.starting_split = split
        self.prosperity = prosperity
        self.num_players = num_players
        self.sandbox = sandbox
