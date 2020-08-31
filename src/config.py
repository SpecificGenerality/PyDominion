from enums import GameConstants, StartingSplit
from constants import BASE_CARDS


class GameConfig:
    def __init__(self, split: StartingSplit, prosperity: bool, numPlayers:int, sandbox=True):
        self.randomizers = BASE_CARDS
        self.kingdomSize = GameConstants.KingdomSize
        self.startingSplit = split
        self.prosperity = prosperity
        self.numPlayers = numPlayers
        self.sandbox = sandbox
