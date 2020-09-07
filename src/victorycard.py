from abc import abstractmethod

from card import Card


class VictoryCard(Card):
    @abstractmethod
    def get_victory_points(self):
        pass

    def get_plus_victory_points(self):
        return 0

    def get_plus_actions(self):
        return 0

    def get_plus_buys(self):
        return 0

    def get_plus_cards(self):
        return 0

    def get_plus_coins(self):
        return 0

    def get_treasure(self):
        return 0

class Estate(VictoryCard):
    def get_coin_cost(self):
        return 2

    def get_victory_points(self):
        return 1

    def __str__(self):
        return "Estate"

class Duchy(VictoryCard):
    def get_coin_cost(self):
        return 5

    def get_victory_points(self):
        return 3

    def __str__(self):
        return "Duchy"

class Province(VictoryCard):
    def get_coin_cost(self):
        return 8

    def get_victory_points(self):
        return 6

    def __str__(self):
        return "Province"

class Colony(VictoryCard):
    def get_coin_cost(self):
        return 11

    def get_victory_points(self):
        return 10

    def __str__(self):
        return "Colony"

class Gardens(VictoryCard):
    def get_coin_cost(self) -> int:
        return 4

    def get_victory_points(self):
        return 0

    def __str__(self):
        return "Gardens"
