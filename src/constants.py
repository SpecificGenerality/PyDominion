from card import Card
from actioncard import *
from cardeffectbase import *

# TODO: Implement Artisan, Sentry, Bandit, Poacher
BASE_EFFECT_MAP = {
    Chapel: ChapelEffect,
    Cellar: CellarEffect,
    Workshop: WorkshopEffect,
    Bureaucrat: BureaucratEffect,
    Militia: MilitiaEffect,
    Moneylender: MoneylenderEffect,
    Remodel: RemodelEffect,
    ThroneRoom: ThroneRoomEffect,
    Library: LibraryEffect,
    Mine: MineEffect,
    Witch: WitchEffect,
}

def getCardEffect(card: Card) -> CardEffect:
    for k, v in BASE_EFFECT_MAP.items():
        if isinstance(card, k):
            return v()
    return None