from card import Card


class TreasureCard(Card):
    def get_plus_victory_points(self) -> int:
        return 0

    def get_victory_points(self) -> int:
        return 0

    def get_plus_actions(self) -> int:
        return 0

    def get_plus_buys(self) -> int:
        return 0

    def get_plus_cards(self) -> int:
        return 0


class Copper(TreasureCard):
    def get_coin_cost(self) -> int:
        return 0

    def get_plus_coins(self) -> int:
        return 1

    def get_treasure(self):
        return 1

    def __str__(self):
        return "Copper"


class Silver(TreasureCard):
    def get_coin_cost(self) -> int:
        return 3

    def get_plus_coins(self) -> int:
        return 2

    def get_treasure(self):
        return 2

    def __str__(self):
        return "Silver"


class Gold(TreasureCard):
    def get_coin_cost(self) -> int:
        return 6

    def get_plus_coins(self) -> int:
        return 3

    def get_treasure(self):
        return 3

    def __str__(self):
        return "Gold"


class Platinum(TreasureCard):
    def get_coin_cost(self) -> int:
        return 9

    def get_plus_coins(self) -> int:
        return 5

    def get_treasure(self):
        return 5

    def __str__(self):
        return "Platinum"
