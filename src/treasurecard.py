from card import Card


class TreasureCard(Card):
    @classmethod
    def get_plus_victory_points(cls) -> int:
        return 0

    @classmethod
    def get_victory_points(cls) -> int:
        return 0

    @classmethod
    def get_plus_actions(cls) -> int:
        return 0

    @classmethod
    def get_plus_buys(cls) -> int:
        return 0

    @classmethod
    def get_plus_cards(cls) -> int:
        return 0


class Copper(TreasureCard):
    @classmethod
    def get_coin_cost(cls) -> int:
        return 0

    @classmethod
    def get_plus_coins(cls) -> int:
        return 1

    @classmethod
    def get_treasure(cls):
        return 1

    def __str__(self):
        return "Copper"


class Silver(TreasureCard):
    @classmethod
    def get_coin_cost(cls) -> int:
        return 3

    @classmethod
    def get_plus_coins(cls) -> int:
        return 2

    @classmethod
    def get_treasure(cls):
        return 2

    def __str__(self):
        return "Silver"


class Gold(TreasureCard):
    @classmethod
    def get_coin_cost(cls) -> int:
        return 6

    @classmethod
    def get_plus_coins(cls) -> int:
        return 3

    @classmethod
    def get_treasure(cls):
        return 3

    def __str__(self):
        return "Gold"


class Platinum(TreasureCard):
    @classmethod
    def get_coin_cost(cls) -> int:
        return 9

    @classmethod
    def get_plus_coins(cls) -> int:
        return 5

    @classmethod
    def get_treasure(cls):
        return 5

    def __str__(self):
        return "Platinum"
