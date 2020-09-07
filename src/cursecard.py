from card import Card

class Curse(Card):
    def get_plus_victory_points(self) -> int:
        return 0

    def get_victory_points(self) -> int:
        return -1

    def get_plus_actions(self) -> int:
        return 0

    def get_plus_buys(self) -> int:
        return 0

    def get_plus_cards(self) -> int:
        return 0

    def get_coin_cost(self) -> int:
        return 0

    def get_plus_coins(self) -> int:
        return 0

    def get_treasure(self) -> int:
        return 0

    def __str__(self):
        return "Curse"

    def __eq__(self, other):
        return isinstance(other, Curse)