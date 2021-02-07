from abc import ABC, abstractmethod


class Card(ABC):
    def __init__(self, copies=1, turns_left=1):
        self.copies = copies
        self.turns_left = turns_left

    @classmethod
    @abstractmethod
    def get_coin_cost(cls) -> int:
        pass

    @classmethod
    @abstractmethod
    def get_victory_points(cls) -> int:
        pass

    @classmethod
    @abstractmethod
    def get_plus_victory_points(cls) -> int:
        pass

    @classmethod
    @abstractmethod
    def get_plus_actions(cls) -> int:
        pass

    @classmethod
    @abstractmethod
    def get_plus_buys(cls) -> int:
        pass

    @classmethod
    @abstractmethod
    def get_plus_cards(cls) -> int:
        pass

    @classmethod
    @abstractmethod
    def get_plus_coins(cls) -> int:
        pass

    @classmethod
    @abstractmethod
    def get_treasure(cls) -> int:
        pass

    def __hash__(self) -> str:
        return hash(str(self))

    def __repr__(self) -> str:
        return str(self)
