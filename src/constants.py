from card import Card
from actioncard import *
from cardeffectbase import *

# TODO: Implement Sentry, Bandit, Poacher
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

def getCardEffect(card: Card) -> CardEffect:
    for k, v in BASE_EFFECT_MAP.items():
        if isinstance(card, k):
            return v()
    return None