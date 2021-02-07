from card import Card


class Curse(Card):
    @classmethod
    def get_plus_victory_points(cls) -> int:
        return 0

    @classmethod
    def get_victory_points(cls) -> int:
        return -1

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
    def get_coin_cost(cls) -> int:
        return 0

    @classmethod
    def get_plus_coins(cls) -> int:
        return 0

    @classmethod
    def get_treasure(cls) -> int:
        return 0

    def __str__(self):
        return "Curse"

    def __eq__(self, other):
        return isinstance(other, Curse)
