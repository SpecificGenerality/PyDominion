from actioncard import *
from cursecard import Curse
from treasurecard import Copper, Gold, Silver
from victorycard import Duchy, Estate, Gardens, Province

'''List of the Base set kingdom cards'''

SANDBOX_CARDS = [Copper, Silver, Gold, Estate, Duchy, Province, Curse]
DEFAULT_KINGDOM = [Chapel, Moat, Village, Militia, Moneylender, Smithy, CouncilRoom, Laboratory, Market, Witch]
UCT_SELECTION = ['ucb1', 'ucb1_tuned', 'robust', 'max', 'secure']

BUY = 25
ACTION = 24

BASE_CARD_NAME = {
        'Artisan': Artisan,
        'Bandit': Bandit,
        'Bureaucrat': Bureaucrat,
        'Cellar': Cellar,
        'Chapel': Chapel,
        'CouncilRoom': CouncilRoom,
        'Festival': Festival,
        'Gardens': Gardens,
        'Harbinger': Harbinger,
        'Laboratory': Laboratory,
        'Library': Library,
        'Market': Market,
        'Merchant': Merchant,
        'Militia': Militia,
        'Mine': Mine,
        'Moat': Moat,
        'Moneylender': Moneylender,
        'Poacher': Poacher,
        'Remodel': Remodel,
        'Sentry': Sentry,
        'Smithy': Smithy,
        'ThroneRoom': ThroneRoom,
        'Vassal': Vassal,
        'Village': Village,
        'Witch': Witch,
        'Workshop': Workshop,
}

BASE_CARDS = [v for v in BASE_CARD_NAME.values()]
