from abc import abstractmethod

from card import Card


class VictoryCard(Card):
    @classmethod
    @abstractmethod
    def get_victory_points(cls):
        pass

    @classmethod
    def get_plus_victory_points(cls):
        return 0

    @classmethod
    def get_plus_actions(cls):
        return 0

    @classmethod
    def get_plus_buys(cls):
        return 0

    @classmethod
    def get_plus_cards(cls):
        return 0

    @classmethod
    def get_plus_coins(cls):
        return 0

    @classmethod
    def get_treasure(cls):
        return 0


class Estate(VictoryCard):
    @classmethod
    def get_coin_cost(cls):
        return 2

    @classmethod
    def get_victory_points(cls):
        return 1

    def __str__(self):
        return "Estate"


class Duchy(VictoryCard):
    @classmethod
    def get_coin_cost(cls):
        return 5

    @classmethod
    def get_victory_points(cls):
        return 3

    def __str__(self):
        return "Duchy"


class Province(VictoryCard):
    @classmethod
    def get_coin_cost(cls):
        return 8

    @classmethod
    def get_victory_points(cls):
        return 6

    def __str__(self):
        return "Province"


class Colony(VictoryCard):
    @classmethod
    def get_coin_cost(cls):
        return 11

    @classmethod
    def get_victory_points(cls):
        return 10

    def __str__(self):
        return "Colony"


class Gardens(VictoryCard):
    @classmethod
    def get_coin_cost(cls) -> int:
        return 4

    @classmethod
    def get_victory_points(cls):
        return 0

    def __str__(self):
        return "Gardens"
