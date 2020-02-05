from enum import Enum, auto, IntEnum

class GameConstants(IntEnum):
    KingdomSize = 10

# TODO: Use auto() after upgrading pygame and running python3.7
class Phase(Enum):
    ActionPhase = auto()
    TreasurePhase = auto()
    BuyPhase = auto()
    # TODO: Extend implementation to use night phase
    # NightPhase = auto()
    CleanupPhase = auto()

class Zone(Enum):
    Discard = auto()
    Deck = auto()
    Hand = auto()
    Trash = auto()
    Play = auto()

class GainZone(Enum):
    GainToDiscard = auto()
    GainToHand = auto()
    GainToDeckTop = auto()
    GainToTrash = auto()

class DiscardZone(Enum):
    DiscardFromHand = auto()
    # E.g. Draw k until <condition>, then discard. (library, saboteur)
    DiscardFromSideZone = auto()
    DiscardFromDeck = auto()

class StartingSplit(Enum):
    Starting25Split = auto()
    Starting34Split = auto()
    StartingRandomSplit = auto()

class DecisionType(Enum):
    DecisionNone = auto()
    DecisionSelectCards = auto()
    DecisionDiscreteChoice = auto()
    DecisionGameOver = auto()

class TriggerState(Enum):
    TriggerNone = auto()
    TriggerProcessed = auto()
    TriggerProcessingRoyalSeal = auto()
    TriggerProcessingWatchtower = auto()

