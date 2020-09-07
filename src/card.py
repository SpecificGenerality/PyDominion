from abc import ABC, abstractmethod


class Card(ABC):
    def __init__(self, copies=1, turns_left=1):
        self.copies = copies
        self.turns_left = turns_left

    @abstractmethod
    def get_coin_cost(self) -> int:
        pass

    @abstractmethod
    def get_victory_points(self) -> int:
        pass

    @abstractmethod
    def get_plus_victory_points(self) -> int:
        pass

    @abstractmethod
    def get_plus_actions(self) -> int:
        pass

    @abstractmethod
    def get_plus_buys(self) -> int:
        pass

    @abstractmethod
    def get_plus_cards(self) -> int:
        pass

    @abstractmethod
    def get_plus_coins(self) -> int:
        pass

    @abstractmethod
    def get_treasure(self) -> int:
        pass

    def __hash__(self) -> str:
        return hash(str(self))

    def __repr__(self) -> str:
        return str(self)
