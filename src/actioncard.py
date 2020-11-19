from card import Card


class ActionCard(Card):
    pass


class BaseActionCard(ActionCard):
    def get_plus_victory_points(self) -> int:
        return 0

    def get_victory_points(self) -> int:
        return 0

    def get_treasure(self):
        return 0


class AttackCard(ActionCard):
    pass


class ReactionCard(ActionCard):
    pass


class Cellar(BaseActionCard):
    def __str__(self):
        return "Cellar"

    def get_coin_cost(self) -> int:
        return 2

    def get_plus_actions(self) -> int:
        return 1

    def get_plus_buys(self) -> int:
        return 0

    def get_plus_cards(self) -> int:
        return 0

    def get_plus_coins(self) -> int:
        return 0


class Chapel(BaseActionCard):
    def get_coin_cost(self) -> int:
        return 2

    def get_plus_actions(self) -> int:
        return 0

    def get_plus_buys(self) -> int:
        return 0

    def get_plus_cards(self) -> int:
        return 0

    def get_plus_coins(self) -> int:
        return 0

    def __str__(self):
        return "Chapel"


class Moat(BaseActionCard, ReactionCard):
    def get_coin_cost(self) -> int:
        return 2

    def get_plus_actions(self) -> int:
        return 0

    def get_plus_buys(self) -> int:
        return 0

    def get_plus_cards(self) -> int:
        return 2

    def get_plus_coins(self) -> int:
        return 0

    def __str__(self):
        return "Moat"


class Harbinger(BaseActionCard):
    def get_coin_cost(self) -> int:
        return 3

    def get_plus_actions(self) -> int:
        return 1

    def get_plus_buys(self) -> int:
        return 0

    def get_plus_cards(self) -> int:
        return 1

    def get_plus_coins(self) -> int:
        return 0

    def __str__(self):
        return "Harbinger"


class Merchant(BaseActionCard):
    def get_coin_cost(self) -> int:
        return 3

    def get_plus_actions(self) -> int:
        return 1

    def get_plus_buys(self) -> int:
        return 0

    def get_plus_cards(self) -> int:
        return 1

    def get_plus_coins(self) -> int:
        return 0

    def __str__(self):
        return "Merchant"


class Vassal(BaseActionCard):
    def get_coin_cost(self) -> int:
        return 3

    def get_plus_actions(self) -> int:
        return 0

    def get_plus_buys(self) -> int:
        return 0

    def get_plus_cards(self) -> int:
        return 0

    def get_plus_coins(self) -> int:
        return 2

    def __str__(self):
        return "Vassal"


class Village(BaseActionCard):
    def get_coin_cost(self) -> int:
        return 3

    def get_plus_actions(self) -> int:
        return 2

    def get_plus_buys(self) -> int:
        return 0

    def get_plus_cards(self) -> int:
        return 1

    def get_plus_coins(self) -> int:
        return 0

    def __str__(self):
        return "Village"


class Workshop(BaseActionCard):
    def get_coin_cost(self) -> int:
        return 3

    def get_plus_actions(self) -> int:
        return 0

    def get_plus_buys(self) -> int:
        return 0

    def get_plus_cards(self) -> int:
        return 0

    def get_plus_coins(self) -> int:
        return 0

    def __str__(self):
        return "Workshop"


class Bureaucrat(BaseActionCard, AttackCard):
    def get_coin_cost(self) -> int:
        return 4

    def get_plus_actions(self) -> int:
        return 0

    def get_plus_buys(self) -> int:
        return 0

    def get_plus_cards(self) -> int:
        return 0

    def get_plus_coins(self) -> int:
        return 0

    def __str__(self):
        return "Bureaucrat"


class Militia(BaseActionCard, AttackCard):
    def get_coin_cost(self) -> int:
        return 4

    def get_plus_actions(self) -> int:
        return 0

    def get_plus_buys(self) -> int:
        return 0

    def get_plus_cards(self) -> int:
        return 0

    def get_plus_coins(self) -> int:
        return 2

    def __str__(self):
        return "Militia"


class Moneylender(BaseActionCard):
    def get_coin_cost(self) -> int:
        return 4

    def get_plus_actions(self) -> int:
        return 0

    def get_plus_buys(self) -> int:
        return 0

    def get_plus_cards(self) -> int:
        return 0

    def get_plus_coins(self) -> int:
        return 0

    def __str__(self):
        return "MoneyLender"


class Poacher(BaseActionCard):
    def get_coin_cost(self) -> int:
        return 4

    def get_plus_actions(self) -> int:
        return 1

    def get_plus_buys(self) -> int:
        return 0

    def get_plus_cards(self) -> int:
        return 1

    def get_plus_coins(self) -> int:
        return 1

    def __str__(self):
        return "Poacher"


class Remodel(BaseActionCard):
    def get_coin_cost(self) -> int:
        return 4

    def get_plus_actions(self) -> int:
        return 0

    def get_plus_buys(self) -> int:
        return 0

    def get_plus_cards(self) -> int:
        return 0

    def get_plus_coins(self) -> int:
        return 0

    def __str__(self):
        return "Remodel"


class Smithy(BaseActionCard):
    def get_coin_cost(self) -> int:
        return 4

    def get_plus_actions(self) -> int:
        return 0

    def get_plus_buys(self) -> int:
        return 0

    def get_plus_cards(self) -> int:
        return 3

    def get_plus_coins(self) -> int:
        return 0

    def __str__(self):
        return "Smithy"


class ThroneRoom(BaseActionCard):
    def get_coin_cost(self) -> int:
        return 4

    def get_plus_actions(self) -> int:
        return 0

    def get_plus_buys(self) -> int:
        return 0

    def get_plus_cards(self) -> int:
        return 0

    def get_plus_coins(self) -> int:
        return 0

    def __str__(self):
        return "Throne Room"


class Bandit(BaseActionCard):
    def get_coin_cost(self) -> int:
        return 5

    def get_plus_actions(self) -> int:
        return 0

    def get_plus_buys(self) -> int:
        return 0

    def get_plus_cards(self) -> int:
        return 0

    def get_plus_coins(self) -> int:
        return 0

    def __str__(self):
        return "Bandit"


class CouncilRoom(BaseActionCard):
    def get_coin_cost(self) -> int:
        return 5

    def get_plus_actions(self) -> int:
        return 0

    def get_plus_cards(self) -> int:
        return 4

    def get_plus_buys(self) -> int:
        return 1

    def get_plus_coins(self) -> int:
        return 0

    def __str__(self):
        return "Council Room"


class Festival(BaseActionCard):
    def get_coin_cost(self) -> int:
        return 5

    def get_plus_actions(self) -> int:
        return 2

    def get_plus_cards(self) -> int:
        return 0

    def get_plus_buys(self) -> int:
        return 1

    def get_plus_coins(self) -> int:
        return 2

    def __str__(self):
        return "Festival"


class Laboratory(BaseActionCard):
    def get_coin_cost(self) -> int:
        return 5

    def get_plus_actions(self) -> int:
        return 1

    def get_plus_cards(self) -> int:
        return 2

    def get_plus_buys(self) -> int:
        return 0

    def get_plus_coins(self) -> int:
        return 0

    def __str__(self):
        return "Laboratory"


class Library(BaseActionCard):
    def get_coin_cost(self) -> int:
        return 5

    def get_plus_actions(self) -> int:
        return 0

    def get_plus_cards(self) -> int:
        return 0

    def get_plus_buys(self) -> int:
        return 0

    def get_plus_coins(self) -> int:
        return 0

    def __str__(self):
        return "Library"


class Market(BaseActionCard):
    def get_coin_cost(self) -> int:
        return 5

    def get_plus_actions(self) -> int:
        return 1

    def get_plus_cards(self) -> int:
        return 1

    def get_plus_buys(self) -> int:
        return 1

    def get_plus_coins(self) -> int:
        return 1

    def __str__(self):
        return "Market"


class Mine(BaseActionCard):
    def get_coin_cost(self) -> int:
        return 5

    def get_plus_actions(self) -> int:
        return 0

    def get_plus_cards(self) -> int:
        return 0

    def get_plus_buys(self) -> int:
        return 0

    def get_plus_coins(self) -> int:
        return 0

    def __str__(self):
        return "Mine"


class Sentry(BaseActionCard):
    def get_coin_cost(self) -> int:
        return 5

    def get_plus_actions(self) -> int:
        return 1

    def get_plus_cards(self) -> int:
        return 1

    def get_plus_buys(self) -> int:
        return 0

    def get_plus_coins(self) -> int:
        return 0

    def __str__(self):
        return "Sentry"


class Witch(BaseActionCard, AttackCard):
    def get_coin_cost(self) -> int:
        return 5

    def get_plus_actions(self) -> int:
        return 0

    def get_plus_cards(self) -> int:
        return 2

    def get_plus_buys(self) -> int:
        return 0

    def get_plus_coins(self) -> int:
        return 0

    def __str__(self):
        return "Witch"


class Artisan(BaseActionCard):
    def get_coin_cost(self) -> int:
        return 6

    def get_plus_actions(self) -> int:
        return 0

    def get_plus_cards(self) -> int:
        return 0

    def get_plus_buys(self) -> int:
        return 0

    def get_plus_coins(self) -> int:
        return 0

    def __str__(self):
        return "Artisan"
