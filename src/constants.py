from actioncard import *
from card import Card
from cursecard import Curse
from victorycard import Gardens

'''List of the Base set kingdom cards'''
# TODO: Implement Vassal, Merchant
BASE_CARDS = [Cellar, Chapel, Moat, \
        Harbinger, Merchant, Village, Workshop, \
        Bureaucrat, Gardens, Militia, Moneylender, Poacher, Remodel, Smithy, ThroneRoom, \
        CouncilRoom, Festival, Laboratory, Library, Market, Mine, Witch, \
        Artisan]

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
