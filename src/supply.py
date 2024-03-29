import logging
from collections.abc import MutableMapping
from random import shuffle
from typing import Dict, List

from card import Card
from config import GameConfig
from cursecard import Curse
from treasurecard import Copper, Gold, Silver
from victorycard import Colony, Duchy, Estate, Province


class Supply(MutableMapping):
    def __init__(self, config: GameConfig, *args, **kwargs):
        # Gardens, Workshop, Laboratory, Village, Witch, Festival, Mine, Chapel, CouncilRoom, Gardens
        def init_kingdom_cards(supply: Dict) -> None:
            must_include = config.must_include
            for i in range(min(config.kingdom_size, len(must_include))):
                supply[must_include[i]] = 10

            shuffle(config.randomizers)
            for i in range(config.kingdom_size - len(must_include)):
                supply[config.randomizers[i]] = 10

        def init_supply(supply: Dict) -> None:
            if config.num_players <= 2:
                supply[Copper] = 46
                supply[Curse] = 10
                supply[Estate] = 8
                supply[Duchy] = 8
                supply[Province] = 8
            elif config.num_players == 3:
                supply[Copper] = 39
                supply[Curse] = 20
                supply[Estate] = 12
                supply[Duchy] = 12
                supply[Province] = 12
            else:
                supply[Copper] = 32
                supply[Curse] = 30
                supply[Estate] = 12
                supply[Duchy] = 12
                supply[Province] = 12
            supply[Silver] = 40
            supply[Gold] = 30

        self._supply = dict(*args, **kwargs)
        init_supply(self._supply)

        if not config.sandbox:
            init_kingdom_cards(self._supply)

        self.update(self._supply)

    @property
    def empty_stack_count(self) -> int:
        return sum(1 if count == 0 else 0 for count in self._supply.values())

    def is_game_over(self) -> bool:
        if Colony in self._supply and self._supply[Colony] == 0:
            logging.info('Game over. Colonies ran out.')
            return True
        elif self._supply[Province] == 0:
            logging.info('Game over. Provinces ran out.')
            return True
        elif self.empty_stack_count >= 3:
            logging.info('Game over. Three supply piles ran out.')
            return True
        return False

    def get_supply_card_types(self) -> List[str]:
        return [str(c()) for c in self._supply.keys()]

    def __getitem__(self, key):
        return self._supply.__getitem__(key)

    def __setitem__(self, key: Card, val: int):
        return self._supply.__setitem__(key, val)

    def __delitem__(self, key):
        return self._supply.__delitem__(key)

    def __iter__(self):
        return self._supply.__iter__()

    def __len__(self):
        return self._supply.__len__()
