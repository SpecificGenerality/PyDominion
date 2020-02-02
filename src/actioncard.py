from abc import ABC
from card import Card

class ActionCard(Card):
    pass

class BaseActionCard(ActionCard):
    def getPlusVictoryPoints(self) -> int:
        return 0

    def getVictoryPoints(self) -> int:
        return 0

    def getTreasure(self):
        return 0

class AttackCard(ActionCard):
    pass

class ReactionCard(ActionCard):
    pass

class Cellar(BaseActionCard):
    def __str__(self):
        return "Cellar"

    def getCoinCost(self) -> int:
        return 2

    def getPlusActions(self) -> int:
        return 1

    def getPlusBuys(self) -> int:
        return 0

    def getPlusCards(self) -> int:
        return 0

    def getPlusCoins(self) -> int:
        return 0

class Chapel(BaseActionCard):
    def __init__(self):
        self.effect = ChapelEffect()

    def getCoinCost(self) -> int:
        return 2

    def getPlusActions(self) -> int:
        return 0

    def getPlusBuys(self) -> int:
        return 0

    def getPlusCards(self) -> int:
        return 0

    def getPlusCoins(self) -> int:
        return 0

    def __str__(self):
        return "Chapel"

class Moat(BaseActionCard, ReactionCard):
    def getCoinCost(self) -> int:
        return 2

    def getPlusActions(self) -> int:
        return 0

    def getPlusBuys(self) -> int:
        return 0

    def getPlusCards(self) -> int:
        return 2

    def getPlusCoins(self) -> int:
        return 0

    def __str__(self):
        return "Moat"

class Harbinger(BaseActionCard):
    def getCoinCost(self) -> int:
        return 3

    def getPlusActions(self) -> int:
        return 1

    def getPlusBuys(self) -> int:
        return 0

    def getPlusCards(self) -> int:
        return 1

    def getPlusCoins(self) -> int:
        return 0

    def __str__(self):
        return "Harbinger"

class Merchant(BaseActionCard):
    def getCoinCost(self) -> int:
        return 3

    def getPlusActions(self) -> int:
        return 1

    def getPlusBuys(self) -> int:
        return 0

    def getPlusCards(self) -> int:
        return 1

    def getPlusCoins(self) -> int:
        return 0

    def __str__(self):
        return "Merchant"

class Vassal(BaseActionCard):
    def getCoinCost(self) -> int:
        return 3

    def getPlusActions(self) -> int:
        return 0

    def getPlusBuys(self) -> int:
        return 0

    def getPlusCards(self) -> int:
        return 0

    def getPlusCoins(self) -> int:
        return 2

    def __str__(self):
        return "Vassal"

class Village(BaseActionCard):
    def getCoinCost(self) -> int:
        return 3

    def getPlusActions(self) -> int:
        return 2

    def getPlusBuys(self) -> int:
        return 0

    def getPlusCards(self) -> int:
        return 1

    def getPlusCoins(self) -> int:
        return 0

    def __str__(self):
        return "Village"

class Workshop(BaseActionCard):
    def __init__(self):
        self.effect = WorkshopEffect()

    def getCoinCost(self) -> int:
        return 3

    def getPlusActions(self) -> int:
        return 0

    def getPlusBuys(self) -> int:
        return 0

    def getPlusCards(self) -> int:
        return 0

    def getPlusCoins(self) -> int:
        return 0

    def __str__(self):
        return "Workshop"

class Bureaucrat(BaseActionCard, AttackCard):
    def __init__(self):
        self.effect = BureaucratEffect()

    def getCoinCost(self) -> int:
        return 4

    def getPlusActions(self) -> int:
        return 0

    def getPlusBuys(self) -> int:
        return 0

    def getPlusCards(self) -> int:
        return 0

    def getPlusCoins(self) -> int:
        return 0

    def __str__(self):
        return "Bureaucrat"

class Militia(BaseActionCard, AttackCard):
    def getCoinCost(self) -> int:
        return 4

    def getPlusActions(self) -> int:
        return 0

    def getPlusBuys(self) -> int:
        return 0

    def getPlusCards(self) -> int:
        return 0

    def getPlusCoins(self) -> int:
        return 2

    def __str__(self):
        return "Militia"

class Moneylender(BaseActionCard):
    def getCoinCost(self) -> int:
        return 4

    def getPlusActions(self) -> int:
        return 0

    def getPlusBuys(self) -> int:
        return 0

    def getPlusCards(self) -> int:
        return 0

    def getPlusCoins(self) -> int:
        return 0

    def __str__(self):
        return "MoneyLender"

class Poacher(BaseActionCard):
    def getCoinCost(self) -> int:
        return 4

    def getPlusActions(self) -> int:
        return 1

    def getPlusBuys(self) -> int:
        return 0

    def getPlusCards(self) -> int:
        return 1

    def getPlusCoins(self) -> int:
        return 1

    def __str__(self):
        return "Poacher"

class Remodel(BaseActionCard):
    def getCoinCost(self) -> int:
        return 4

    def getPlusActions(self) -> int:
        return 0

    def getPlusBuys(self) -> int:
        return 0

    def getPlusCards(self) -> int:
        return 0

    def getPlusCoins(self) -> int:
        return 0

    def __str__(self):
        return "Remodel"

class Smithy(BaseActionCard):
    def getCoinCost(self) -> int:
        return 4

    def getPlusActions(self) -> int:
        return 0

    def getPlusBuys(self) -> int:
        return 0

    def getPlusCards(self) -> int:
        return 3

    def getPlusCoins(self) -> int:
        return 0

    def __str__(self):
        return "Smithy"

class ThroneRoom(BaseActionCard):
    def getCoinCost(self) -> int:
        return 4

    def getPlusActions(self) -> int:
        return 0

    def getPlusBuys(self) -> int:
        return 0

    def getPlusCards(self) -> int:
        return 0

    def getPlusCoins(self) -> int:
        return 0

    def __str__(self):
        return "Throne Room"

class Bandit(BaseActionCard):
    def getCoinCost(self) -> int:
        return 5

    def getPlusActions(self) -> int:
        return 0

    def getPlusBuys(self) -> int:
        return 0

    def getPlusCards(self) -> int:
        return 0

    def getPlusCoins(self) -> int:
        return 0

    def __str__(self):
        return "Bandit"

class CouncilRoom(BaseActionCard):
    def getCoinCost(self) -> int:
        return 5

    def getPlusActions(self) -> int:
        return 0

    def getPlusCards(self) -> int:
        return 4

    def getPlusBuys(self) -> int:
        return 1

    def getPlusCoins(self) -> int:
        return 0

    def __str__(self):
        return "Council Room"

class Festival(BaseActionCard):
    def getCoinCost(self) -> int:
        return 5

    def getPlusActions(self) -> int:
        return 2

    def getPlusCards(self) -> int:
        return 0

    def getPlusBuys(self) -> int:
        return 1

    def getPlusCoins(self) -> int:
        return 2

    def __str__(self):
        return "Festival"

class Laboratory(BaseActionCard):
    def getCoinCost(self) -> int:
        return 5

    def getPlusActions(self) -> int:
        return 1

    def getPlusCards(self) -> int:
        return 2

    def getPlusBuys(self) -> int:
        return 0

    def getPlusCoins(self) -> int:
        return 0

    def __str__(self):
        return "Laboratory"

class Library(BaseActionCard):
    def getCoinCost(self) -> int:
        return 5

    def getPlusActions(self) -> int:
        return 0

    def getPlusCards(self) -> int:
        return 0

    def getPlusBuys(self) -> int:
        return 0

    def getPlusCoins(self) -> int:
        return 0

    def __str__(self):
        return "Library"

class Market(BaseActionCard):
    def getCoinCost(self) -> int:
        return 5

    def getPlusActions(self) -> int:
        return 1

    def getPlusCards(self) -> int:
        return 1

    def getPlusBuys(self) -> int:
        return 1

    def getPlusCoins(self) -> int:
        return 1

    def __str__(self):
        return "Market"

class Mine(BaseActionCard):
    def getCoinCost(self) -> int:
        return 5

    def getPlusActions(self) -> int:
        return 0

    def getPlusCards(self) -> int:
        return 0

    def getPlusBuys(self) -> int:
        return 0

    def getPlusCoins(self) -> int:
        return 0

    def __str__(self):
        return "Mine"

class Sentry(BaseActionCard):
    def getCoinCost(self) -> int:
        return 5

    def getPlusActions(self) -> int:
        return 1

    def getPlusCards(self) -> int:
        return 1

    def getPlusBuys(self) -> int:
        return 0

    def getPlusCoins(self) -> int:
        return 0

    def __str__(self):
        return "Sentry"

class Witch(BaseActionCard, AttackCard):
    def getCoinCost(self) -> int:
        return 5

    def getPlusActions(self) -> int:
        return 0

    def getPlusCards(self) -> int:
        return 2

    def getPlusBuys(self) -> int:
        return 0

    def getPlusCoins(self) -> int:
        return 0

    def __str__(self):
        return "Witch"

class Artisan(BaseActionCard):
    def getCoinCost(self) -> int:
        return 6

    def getPlusActions(self) -> int:
        return 0

    def getPlusCards(self) -> int:
        return 0

    def getPlusBuys(self) -> int:
        return 0

    def getPlusCoins(self) -> int:
        return 0

    def __str__(self):
        return "Artisan"
