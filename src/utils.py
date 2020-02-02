from card import *
from treasurecard import *
from cursecard import *
from victorycard import *
from actioncard import *
from typing import List

def getBaseKingdomCards() -> List:
    return [Cellar, Chapel, Moat, Harbinger, Merchant, \
        Vassal, Village, Workshop, Bureaucrat, Gardens, \
        Militia, Moneylender, Poacher, Remodel, Smithy, \
        ThroneRoom, Bandit, CouncilRoom, Festival, Laboratory, \
        Library, Market, Mine, Sentry, Witch, Artisan]
