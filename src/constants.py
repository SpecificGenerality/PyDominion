from actioncard import *
from card import Card
from cursecard import Curse
from victorycard import Gardens

'''List of the Base set kingdom cards'''
# TODO: Implement Bandit, Vassal, Merchant
BASE_CARDS = [Cellar, Chapel, Moat, \
        Harbinger, Merchant, Village, Workshop, \
        Bureaucrat, Gardens, Militia, Moneylender, Poacher, Remodel, Smithy, ThroneRoom, \
        CouncilRoom, Festival, Laboratory, Library, Market, Mine, Witch, \
        Artisan]

BASE_CARD_NAME = {
        'Cellar': Cellar,
        'Chapel': Chapel,
        'Moat': Moat,
        'Harbinger': Harbinger,
        'Merchant': Merchant,
        'Village': Village,
        'Workshop': Workshop,
        'Bureaucrat': Bureaucrat,
        'Gardens': Gardens,
        'Militia': Militia,
        'Moneylender': Moneylender,
        'Poacher': Poacher,
        'Remodel': Remodel,
        'Smithy': Smithy,
        'ThroneRoom': ThroneRoom,
        'CouncilRoom': CouncilRoom,
        'Festival': Festival,
        'Laboratory': Laboratory,
        'Library': Library,
        'Market': Market,
        'Mine': Mine,
        'Witch': Witch,
        'Artisan': Artisan,
        'Sentry': Sentry,
        'Bandit': Bandit
}
