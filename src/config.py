from enums import StartingSplit, GameConstants
from utils import getBaseKingdomCards

class GameConfig:
    def __init__(self, split: StartingSplit, prosperity: bool, numPlayers:int, sandbox=True):
        self.randomizers = getBaseKingdomCards()
        self.kingdomSize = GameConstants.KingdomSize
        self.startingSplit = split
        self.prosperity = prosperity
        self.numPlayers = numPlayers
        self.sandbox = sandbox