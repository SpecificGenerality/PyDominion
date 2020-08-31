from actioncard import *
from card import Card
from cardeffectbase import *
from cursecard import *
from treasurecard import *
from victorycard import *

# TODO: Implement Sentry, Bandit
BASE_EFFECT_MAP = {
    Artisan: ArtisanEffect,
    Chapel: ChapelEffect,
    Cellar: CellarEffect,
    Harbinger: HarbingerEffect,
    Workshop: WorkshopEffect,
    Bureaucrat: BureaucratEffect,
    Militia: MilitiaEffect,
    Moneylender: MoneylenderEffect,
    Remodel: RemodelEffect,
    ThroneRoom: ThroneRoomEffect,
    Library: LibraryEffect,
    Mine: MineEffect,
    Poacher: PoacherEffect,
    Witch: WitchEffect,
    Gardens: GardensEffect
}

'''List of the Base set kingdom cards'''
# TODO: Implement Sentry, Bandit, Vassal, Merchant
BASE_CARDS = [Cellar, Chapel, Moat, \
        Harbinger, Merchant, Village, Workshop, \
        Bureaucrat, Gardens, Militia, Moneylender, Poacher, Remodel, Smithy, ThroneRoom, \
        CouncilRoom, Festival, Laboratory, Library, Market, Mine, Witch, \
        Artisan]


def getCardEffect(card: Card) -> CardEffect:
    for k, v in BASE_EFFECT_MAP.items():
        if isinstance(card, k):
            return v()
    return None
