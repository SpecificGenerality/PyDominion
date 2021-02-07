from card import Card


class ActionCard(Card):
    pass


class BaseActionCard(ActionCard):
    @classmethod
    def get_plus_victory_points(cls) -> int:
        return 0

    @classmethod
    def get_victory_points(cls) -> int:
        return 0

    @classmethod
    def get_treasure(cls):
        return 0


class AttackCard(ActionCard):
    pass


class ReactionCard(ActionCard):
    pass


class Cellar(BaseActionCard):
    @classmethod
    def get_coin_cost(cls) -> int:
        return 2

    @classmethod
    def get_plus_actions(cls) -> int:
        return 1

    @classmethod
    def get_plus_buys(cls) -> int:
        return 0

    @classmethod
    def get_plus_cards(cls) -> int:
        return 0

    @classmethod
    def get_plus_coins(cls) -> int:
        return 0

    def __str__(self):
        return "Cellar"


class Chapel(BaseActionCard):
    @classmethod
    def get_coin_cost(cls) -> int:
        return 2

    @classmethod
    def get_plus_actions(cls) -> int:
        return 0

    @classmethod
    def get_plus_buys(cls) -> int:
        return 0

    @classmethod
    def get_plus_cards(cls) -> int:
        return 0

    @classmethod
    def get_plus_coins(cls) -> int:
        return 0

    def __str__(self):
        return "Chapel"


class Moat(BaseActionCard, ReactionCard):
    @classmethod
    def get_coin_cost(cls) -> int:
        return 2

    @classmethod
    def get_plus_actions(cls) -> int:
        return 0

    @classmethod
    def get_plus_buys(cls) -> int:
        return 0

    @classmethod
    def get_plus_cards(cls) -> int:
        return 2

    @classmethod
    def get_plus_coins(cls) -> int:
        return 0

    def __str__(self):
        return "Moat"


class Harbinger(BaseActionCard):
    @classmethod
    def get_coin_cost(cls) -> int:
        return 3

    @classmethod
    def get_plus_actions(cls) -> int:
        return 1

    @classmethod
    def get_plus_buys(cls) -> int:
        return 0

    @classmethod
    def get_plus_cards(cls) -> int:
        return 1

    @classmethod
    def get_plus_coins(cls) -> int:
        return 0

    def __str__(self):
        return "Harbinger"


class Merchant(BaseActionCard):
    @classmethod
    def get_coin_cost(cls) -> int:
        return 3

    @classmethod
    def get_plus_actions(cls) -> int:
        return 1

    @classmethod
    def get_plus_buys(cls) -> int:
        return 0

    @classmethod
    def get_plus_cards(cls) -> int:
        return 1

    @classmethod
    def get_plus_coins(cls) -> int:
        return 0

    def __str__(self):
        return "Merchant"


class Vassal(BaseActionCard):
    @classmethod
    def get_coin_cost(cls) -> int:
        return 3

    @classmethod
    def get_plus_actions(cls) -> int:
        return 0

    @classmethod
    def get_plus_buys(cls) -> int:
        return 0

    @classmethod
    def get_plus_cards(cls) -> int:
        return 0

    @classmethod
    def get_plus_coins(cls) -> int:
        return 2

    def __str__(self):
        return "Vassal"


class Village(BaseActionCard):
    @classmethod
    def get_coin_cost(cls) -> int:
        return 3

    @classmethod
    def get_plus_actions(cls) -> int:
        return 2

    @classmethod
    def get_plus_buys(cls) -> int:
        return 0

    @classmethod
    def get_plus_cards(cls) -> int:
        return 1

    @classmethod
    def get_plus_coins(cls) -> int:
        return 0

    def __str__(self):
        return "Village"


class Workshop(BaseActionCard):
    @classmethod
    def get_coin_cost(cls) -> int:
        return 3

    @classmethod
    def get_plus_actions(cls) -> int:
        return 0

    @classmethod
    def get_plus_buys(cls) -> int:
        return 0

    @classmethod
    def get_plus_cards(cls) -> int:
        return 0

    @classmethod
    def get_plus_coins(cls) -> int:
        return 0

    def __str__(self):
        return "Workshop"


class Bureaucrat(BaseActionCard, AttackCard):
    @classmethod
    def get_coin_cost(cls) -> int:
        return 4

    @classmethod
    def get_plus_actions(cls) -> int:
        return 0

    @classmethod
    def get_plus_buys(cls) -> int:
        return 0

    @classmethod
    def get_plus_cards(cls) -> int:
        return 0

    @classmethod
    def get_plus_coins(cls) -> int:
        return 0

    def __str__(self):
        return "Bureaucrat"


class Militia(BaseActionCard, AttackCard):
    @classmethod
    def get_coin_cost(cls) -> int:
        return 4

    @classmethod
    def get_plus_actions(cls) -> int:
        return 0

    @classmethod
    def get_plus_buys(cls) -> int:
        return 0

    @classmethod
    def get_plus_cards(cls) -> int:
        return 0

    @classmethod
    def get_plus_coins(cls) -> int:
        return 2

    def __str__(self):
        return "Militia"


class Moneylender(BaseActionCard):
    @classmethod
    def get_coin_cost(cls) -> int:
        return 4

    @classmethod
    def get_plus_actions(cls) -> int:
        return 0

    @classmethod
    def get_plus_buys(cls) -> int:
        return 0

    @classmethod
    def get_plus_cards(cls) -> int:
        return 0

    @classmethod
    def get_plus_coins(cls) -> int:
        return 0

    def __str__(self):
        return "MoneyLender"


class Poacher(BaseActionCard):
    @classmethod
    def get_coin_cost(cls) -> int:
        return 4

    @classmethod
    def get_plus_actions(cls) -> int:
        return 1

    @classmethod
    def get_plus_buys(cls) -> int:
        return 0

    @classmethod
    def get_plus_cards(cls) -> int:
        return 1

    @classmethod
    def get_plus_coins(cls) -> int:
        return 1

    def __str__(self):
        return "Poacher"


class Remodel(BaseActionCard):
    @classmethod
    def get_coin_cost(cls) -> int:
        return 4

    @classmethod
    def get_plus_actions(cls) -> int:
        return 0

    @classmethod
    def get_plus_buys(cls) -> int:
        return 0

    @classmethod
    def get_plus_cards(cls) -> int:
        return 0

    @classmethod
    def get_plus_coins(cls) -> int:
        return 0

    def __str__(self):
        return "Remodel"


class Smithy(BaseActionCard):
    @classmethod
    def get_coin_cost(cls) -> int:
        return 4

    @classmethod
    def get_plus_actions(cls) -> int:
        return 0

    @classmethod
    def get_plus_buys(cls) -> int:
        return 0

    @classmethod
    def get_plus_cards(cls) -> int:
        return 3

    @classmethod
    def get_plus_coins(cls) -> int:
        return 0

    def __str__(self):
        return "Smithy"


class ThroneRoom(BaseActionCard):
    @classmethod
    def get_coin_cost(cls) -> int:
        return 4

    @classmethod
    def get_plus_actions(cls) -> int:
        return 0

    @classmethod
    def get_plus_buys(cls) -> int:
        return 0

    @classmethod
    def get_plus_cards(cls) -> int:
        return 0

    @classmethod
    def get_plus_coins(cls) -> int:
        return 0

    def __str__(self):
        return "Throne Room"


class Bandit(BaseActionCard):
    @classmethod
    def get_coin_cost(cls) -> int:
        return 5

    @classmethod
    def get_plus_actions(cls) -> int:
        return 0

    @classmethod
    def get_plus_buys(cls) -> int:
        return 0

    @classmethod
    def get_plus_cards(cls) -> int:
        return 0

    @classmethod
    def get_plus_coins(cls) -> int:
        return 0

    def __str__(self):
        return "Bandit"


class CouncilRoom(BaseActionCard):
    @classmethod
    def get_coin_cost(cls) -> int:
        return 5

    @classmethod
    def get_plus_actions(cls) -> int:
        return 0

    @classmethod
    def get_plus_cards(cls) -> int:
        return 4

    @classmethod
    def get_plus_buys(cls) -> int:
        return 1

    @classmethod
    def get_plus_coins(cls) -> int:
        return 0

    def __str__(self):
        return "Council Room"


class Festival(BaseActionCard):
    @classmethod
    def get_coin_cost(cls) -> int:
        return 5

    @classmethod
    def get_plus_actions(cls) -> int:
        return 2

    @classmethod
    def get_plus_cards(cls) -> int:
        return 0

    @classmethod
    def get_plus_buys(cls) -> int:
        return 1

    @classmethod
    def get_plus_coins(cls) -> int:
        return 2

    def __str__(self):
        return "Festival"


class Laboratory(BaseActionCard):
    @classmethod
    def get_coin_cost(cls) -> int:
        return 5

    @classmethod
    def get_plus_actions(cls) -> int:
        return 1

    @classmethod
    def get_plus_cards(cls) -> int:
        return 2

    @classmethod
    def get_plus_buys(cls) -> int:
        return 0

    @classmethod
    def get_plus_coins(cls) -> int:
        return 0

    def __str__(self):
        return "Laboratory"


class Library(BaseActionCard):
    @classmethod
    def get_coin_cost(cls) -> int:
        return 5

    @classmethod
    def get_plus_actions(cls) -> int:
        return 0

    @classmethod
    def get_plus_cards(cls) -> int:
        return 0

    @classmethod
    def get_plus_buys(cls) -> int:
        return 0

    @classmethod
    def get_plus_coins(cls) -> int:
        return 0

    def __str__(self):
        return "Library"


class Market(BaseActionCard):
    @classmethod
    def get_coin_cost(cls) -> int:
        return 5

    @classmethod
    def get_plus_actions(cls) -> int:
        return 1

    @classmethod
    def get_plus_cards(cls) -> int:
        return 1

    @classmethod
    def get_plus_buys(cls) -> int:
        return 1

    @classmethod
    def get_plus_coins(cls) -> int:
        return 1

    def __str__(self):
        return "Market"


class Mine(BaseActionCard):
    @classmethod
    def get_coin_cost(cls) -> int:
        return 5

    @classmethod
    def get_plus_actions(cls) -> int:
        return 0

    @classmethod
    def get_plus_cards(cls) -> int:
        return 0

    @classmethod
    def get_plus_buys(cls) -> int:
        return 0

    @classmethod
    def get_plus_coins(cls) -> int:
        return 0

    def __str__(self):
        return "Mine"


class Sentry(BaseActionCard):
    @classmethod
    def get_coin_cost(cls) -> int:
        return 5

    @classmethod
    def get_plus_actions(cls) -> int:
        return 1

    @classmethod
    def get_plus_cards(cls) -> int:
        return 1

    @classmethod
    def get_plus_buys(cls) -> int:
        return 0

    @classmethod
    def get_plus_coins(cls) -> int:
        return 0

    def __str__(self):
        return "Sentry"


class Witch(BaseActionCard, AttackCard):
    @classmethod
    def get_coin_cost(cls) -> int:
        return 5

    @classmethod
    def get_plus_actions(cls) -> int:
        return 0

    @classmethod
    def get_plus_cards(cls) -> int:
        return 2

    @classmethod
    def get_plus_buys(cls) -> int:
        return 0

    @classmethod
    def get_plus_coins(cls) -> int:
        return 0

    def __str__(self):
        return "Witch"


class Artisan(BaseActionCard):
    @classmethod
    def get_coin_cost(cls) -> int:
        return 6

    @classmethod
    def get_plus_actions(cls) -> int:
        return 0

    @classmethod
    def get_plus_cards(cls) -> int:
        return 0

    @classmethod
    def get_plus_buys(cls) -> int:
        return 0

    @classmethod
    def get_plus_coins(cls) -> int:
        return 0

    def __str__(self):
        return "Artisan"
