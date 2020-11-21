import logging
from abc import ABC, abstractmethod
from collections import Counter
from typing import List, Union

import torch

from actioncard import ActionCard, Merchant, Moat
from card import Card
from config import GameConfig
from cursecard import Curse
from enums import (DecisionType, DiscardZone, GainZone, Phase, TriggerState,
                   Zone)
from playerstate import PlayerState
from supply import Supply
from treasurecard import Copper, Silver, TreasureCard
from utils import remove_card, remove_first_card
from victorycard import VictoryCard


class DecisionResponse:
    def __init__(self, cards: List[Card], choice=-1):
        self.cards = cards
        self.choice = choice
        self.single_card = None

    def __str__(self) -> str:
        return f'Choices: {self.cards}\nChoice: {self.choice}\nSingle Card: {self.single_card}'


class DecisionState:
    def __init__(self):
        self.type = DecisionType.DecisionNone
        self.card_choices = []
        self.active_card = None
        self.min_cards = None
        self.max_cards = None
        self.text = None
        self.controlling_player = None

    def is_trivial(self) -> bool:
        return (self.type == DecisionType.DecisionSelectCards and len(self.card_choices) == 1 and self.min_cards == 1 and self.max_cards == 1) or (self.type == DecisionType.DecisionDiscreteChoice and self.min_cards == self.max_cards == 1)

    def trivial_response(self) -> DecisionResponse:
        return DecisionResponse([self.card_choices[0]]) if self.type == DecisionType.DecisionSelectCards else DecisionResponse([], 0)

    def select_cards(self, card: Card, min_cards: int, max_cards: int):
        self.active_card = card
        self.type = DecisionType.DecisionSelectCards
        self.min_cards = min_cards
        self.max_cards = max_cards
        self.controlling_player = -1
        self.card_choices = []

    def make_discrete_choice(self, card: Card, option_count: int):
        self.active_card = card
        self.type = DecisionType.DecisionDiscreteChoice
        self.min_cards = option_count
        self.max_cards = option_count
        self.controlling_player = -1

    def add_unique_card(self, card: Card):
        if card not in self.card_choices:
            self.card_choices.append(card)

    def add_unique_cards(self, cardList: List[Card]):
        for card in cardList:
            self.add_unique_card(card)

    def gain_card_from_supply(self, s, card: Card, min_cost: int, max_cost: int):
        self.select_cards(card, 1, 1)
        # TODO: Refactor
        for k, v in s.supply.items():
            supply_card = k()
            cost = s.get_supply_cost(supply_card)
            if v > 0 and cost >= min_cost and cost <= max_cost:
                self.add_unique_card(supply_card)

    def gain_treasure_from_supply(self, s, card: Card, min_cost: int, max_cost: int):
        self.select_cards(card, 1, 1)
        self.text = 'Select a treasure to gain:'

        # TODO: Refactor
        for k, v in s.supply.items():
            supplyCard = k()
            cost = s.get_supply_cost(supplyCard)
            if v > 0 and cost >= min_cost and cost <= max_cost:
                self.add_unique_card(supplyCard)

    def print_card_choices(self):
        for i, card in enumerate(self.card_choices):
            logging.info(f'{i}: {card}')


class StateFeature:
    def __init__(self, config: GameConfig, supply: Supply, player_states: List[PlayerState]):
        self.num_cards = len(supply)
        self.num_players = config.num_players
        # Hand/play, deck, discard
        self.num_zones = 3
        self.idxs = dict([(str(k()), i) for i, k in enumerate(supply.keys())])

        # Offsets within player subfeature
        self.deck_offset = 0
        self.hand_offset = 1
        self.discard_offset = 2

        # +1 for supply
        self.feature = torch.zeros((self.num_players * self.num_zones + 1) * self.num_cards, device='cuda')

        # Fill supply card counts
        for k, v in supply.items():
            idx = self.idxs[str(k())]
            self.feature[idx] = v

        # Fill player card counts
        for i, p_state in enumerate(player_states):
            offset = self.num_cards + i * self.num_zones * self.num_cards + self.deck_offset * self.num_cards
            for card, count in p_state.get_card_counts().items():
                idx = self.idxs[str(card)]
                self.feature[idx + offset] = count

    def get_player_idx(self, player: int) -> int:
        return self.num_cards + player * self.num_zones * self.num_cards

    def get_zone_idx(self, player: int, zone: Union[Zone, DiscardZone, GainZone]) -> int:
        p_offset = self.get_player_idx(player)

        if zone == Zone.Deck or zone == DiscardZone.DiscardFromDeck or zone == GainZone.GainToDeckTop:
            z_offset_idx = self.deck_offset
        elif zone == Zone.Discard or zone == GainZone.GainToDiscard:
            z_offset_idx = self.discard_offset
        elif zone == Zone.Hand or zone == Zone.Play or zone == DiscardZone.DiscardFromHand or zone == GainZone.GainToHand:
            z_offset_idx = self.hand_offset
        else:
            raise ValueError(f'{zone} not supported.')

        return p_offset + z_offset_idx * self.num_cards

    def get_card_offset(self, card: Card) -> int:
        return self.idxs[str(card)]

    # TODO: Account for imperfect information
    def get_player_feature(self, player: int) -> torch.tensor:
        return self.feature

    def _dec_inc(self, src: int, tgt: int) -> None:
        self.feature[src] = self.feature[src] - 1
        self.feature[tgt] = self.feature[tgt] + 1

    def _mov_zero(self, src: int, tgt: int, length: int) -> None:
        self.feature[tgt:tgt + length] = self.feature[tgt:tgt + length] + self.feature[src:src + length]
        self.feature[src:src + length] = 0

    def shuffle(self, player: int) -> None:
        deck_idx = self.get_zone_idx(player, Zone.Deck)
        discard_idx = self.get_zone_idx(player, Zone.Discard)

        self._mov_zero(discard_idx, deck_idx, self.num_cards)

    def draw_card(self, player: int, card: Card) -> None:
        self.move_card(player, card, Zone.Deck, Zone.Hand)

    def discard_card(self, player: int, card: Card, zone: DiscardZone) -> None:
        self.move_card(player, card, zone, Zone.Discard)

    def discard_hand(self, player: int) -> None:
        src_idx = self.get_zone_idx(player, Zone.Hand)
        tgt_idx = self.get_zone_idx(player, Zone.Discard)

        self._mov_zero(src_idx, tgt_idx, self.num_cards)

    def gain_card(self, player: int, card: Card, zone: GainZone) -> None:
        src_offset = self.get_zone_idx(player, zone)
        card_idx = self.idxs[str(card)]

        self.feature[src_offset + card_idx] = self.feature[src_offset + card_idx] + 1
        self.feature[card_idx] = self.feature[card_idx] - 1

    def move_card(self, player: int, card: Card, src_zone: Union[Zone, GainZone, DiscardZone], dest_zone: Union[Zone, GainZone, DiscardZone]):
        src_base_idx = self.get_zone_idx(player, src_zone)
        dest_base_idx = self.get_zone_idx(player, dest_zone)

        card_offset = self.idxs[str(card)]

        src_idx = src_base_idx + card_offset
        dest_idx = dest_base_idx + card_offset

        self._dec_inc(src_idx, dest_idx)

    # TODO: Update to split hand/play
    def play_card(self, player: int, card: Card, zone: Zone) -> None:
        return

    # TODO: Update to allow duration cards
    def update_play_area(self, player: int) -> None:
        src_idx = self.get_zone_idx(player, Zone.Play)
        tgt_idx = self.get_zone_idx(player, Zone.Discard)

        self._mov_zero(src_idx, tgt_idx, self.num_cards)

    def trash_card(self, player: int, card: Card, zone: Zone) -> None:
        src_offset = self.get_zone_idx(player, zone)
        src_idx = src_offset + self.idxs[str(card)]

        self.feature[src_idx] = self.feature[src_idx] - 1

    def __len__(self) -> int:
        return self.feature.__len__()


class State:
    def __init__(self, config: GameConfig):
        self.players = [i for i in range(config.num_players)]
        self.player_states = [PlayerState(config) for i in range(config.num_players)]
        self.phase = Phase.ActionPhase
        self.decision = DecisionState()
        self.player = 0
        self.supply = Supply(config)
        self.trash = []
        self.events = []
        self.feature = StateFeature(config, self.supply, self.player_states)

    def draw_card(self, player: int) -> None:
        p_state: PlayerState = self.player_states[player]
        if not p_state._deck:
            self.shuffle(player)

        if not p_state._deck:
            logging.info(f'Player {player} tries to draw but has no cards')
        else:
            card = p_state.draw_card()
            self.feature.draw_card(player, card)

    def draw_hand(self, player: int):
        num_cards_to_draw = 5
        for _ in range(num_cards_to_draw):
            self.draw_card(player)
        logging.info(f'Player {player} draws a new hand.')

    def discard_card(self, player: int, card: Card, zone: DiscardZone):
        p_state: PlayerState = self.player_states[player]
        p_state.discard_card(card, zone)
        self.feature.discard_card(player, card, zone)

    def discard_hand(self, player: int):
        p_state: PlayerState = self.player_states[player]
        p_state.discard_hand()
        self.feature.discard_hand(player)
        logging.info(f'Player {player} discards their hand')

    def gain_card(self, player: int, card: Card, zone: GainZone, bought: bool):
        p_state: PlayerState = self.player_states[player]

        if self.supply[type(card)] > 0:
            p_state.gain_card(card, zone)
            self.feature.gain_card(player, card, zone)

            self.supply[type(card)] -= 1

            if bought:
                cost = self.get_supply_cost(card)
                logging.info(f'Player {player} spends {cost} and buys {card}')
                p_state.coins -= cost
                p_state.buys -= 1
        else:
            if bought:
                logging.info(f'Player {player} cannot buy {card}')
                p_state.buys -= 1
            else:
                logging.info(f'Player {player} cannot gain {card}')

    def move_card(self, player: int, card: Card, src_zone: Zone, dest_zone: Zone):
        p_state: PlayerState = self.player_states[self.player]
        p_state.move_card(card, src_zone, dest_zone)
        self.feature.move_card(card, src_zone, dest_zone)

    def shuffle(self, player: int) -> None:
        p_state: PlayerState = self.player_states[player]
        p_state.shuffle()
        self.feature.shuffle(player)

    def trash_card(self, card: Card, zone: Zone, player: int) -> None:
        p_state: PlayerState = self.player_states[player]

        trashed_card = p_state.trash_card(card, zone)

        if trashed_card:
            self.trash.append(trashed_card)
            self.feature.trash_card(player, trashed_card, zone)
            logging.info(f'Player {player} trashes {card} from {zone}.')
        else:
            logging.info(f'Player {player} {zone} is empty, trashing nothing.')

    def update_play_area(self, player: int):
        p_state: PlayerState = self.player_states[player]
        p_state.update_play_area()
        self.feature.update_play_area(player)

    def play_card(self, player: int, card: Card, zone: Zone = Zone.Hand) -> None:
        p_state: PlayerState = self.player_states[player]
        p_state.play_card(card, zone)
        self.feature.play_card(player, card, zone)

    def process_action(self, card: Card):
        import cardeffectbase
        p_state: PlayerState = self.player_states[self.player]
        p_state.actions += card.get_plus_actions()
        p_state.buys += card.get_plus_buys()
        p_state.coins += card.get_plus_coins()

        for _ in range(card.get_plus_cards()):
            self.draw_card(self.player)

        effect = cardeffectbase.get_card_effect(card)
        if effect:
            effect.play_action(self)

    def process_treasure(self, card: Card):
        p_state: PlayerState = self.player_states[self.player]

        def get_merchant_coins() -> int:
            if isinstance(card, Silver) and sum([1 if isinstance(c, Silver) else 0 for c in p_state._play_area]) == 1:
                return sum([card.copies if isinstance(card, Merchant) else 0 for card in p_state._play_area])
            else:
                return 0

        import cardeffectbase

        assert isinstance(card, TreasureCard), 'Attemped to processTreasure a non-treasure card'
        treasureValue = card.get_treasure()
        merchantCoins = get_merchant_coins()

        logging.info(f'Player {self.player} gets ${treasureValue}')

        p_state.coins += treasureValue
        p_state.coins += merchantCoins

        if merchantCoins > 0:
            logging.info(f'Player {self.player} gets ${merchantCoins} from Merchant')
        p_state.buys += card.get_plus_buys()

        effect = cardeffectbase.get_card_effect(card)
        if effect:
            effect.play_action(self)

    def process_decision(self, response: DecisionResponse):
        if self.decision.type == DecisionType.DecisionGameOver:
            return
        if self.decision.type == DecisionType.DecisionNone:
            raise ValueError('No decision active')

        single_card = response.single_card
        if self.decision.type == DecisionType.DecisionSelectCards and self.decision.max_cards <= 1:
            if len(response.cards) == 1:
                single_card = response.cards[0]
                # assert self.decision.min_cards == 0 or single_card != None, 'No response chosen'
            else:
                # do some asserts here
                pass

        self.decision.type = DecisionType.DecisionNone
        p = self.player_states[self.player]

        if not self.decision.active_card:
            if self.phase == Phase.ActionPhase:
                if single_card is None:
                    logging.info(f'Player {self.player} chooses not to play an action')
                    self.phase = Phase.TreasurePhase
                else:
                    logging.info(f'Playing {single_card}')
                    self.play_card(self.player, single_card)
                    p.actions -= 1
                    self.process_action(single_card)
            elif self.phase == Phase.TreasurePhase:
                if single_card is None:
                    logging.info(f'Player {self.player} chooses not to play a treasure')
                    self.phase = Phase.BuyPhase
                else:
                    self.play_card(self.player, single_card)
                    self.process_treasure(single_card)
            elif self.phase == Phase.BuyPhase:
                if single_card is None:
                    logging.info(f'Player {self.player} chooses not to buy a card')
                    self.phase = Phase.CleanupPhase
                else:
                    self.events.append(GainCard(GainZone.GainToDiscard, self.player, single_card, True, False))
        else:
            import cardeffectbase
            activeCardEffect = cardeffectbase.get_card_effect(self.decision.active_card)
            if activeCardEffect and activeCardEffect.can_process_decisions():
                activeCardEffect.process_decision(self, response)
            elif len(self.events) > 0 and self.events[-1].can_process_decisions():
                self.events[-1].process_decision(self, response)
            else:
                logging.error(f'Decision from {self.decision.active_card} cannot be processed')

        if self.decision.controlling_player == -1:
            self.decision.controlling_player = self.player
        self.decision.max_cards = min(self.decision.max_cards, len(self.decision.card_choices))

    # TODO: peddler, bridge, quarry, and plunder affect this method
    def get_supply_cost(self, card: Card) -> int:
        return card.get_coin_cost()

    def get_player_score(self, player: int) -> int:
        import cardeffectbase
        p_state: PlayerState = self.player_states[player]
        score = 0
        allCards = p_state.cards

        for card in allCards:
            score += card.get_victory_points()
            effect = cardeffectbase.get_card_effect(card)
            if isinstance(card, VictoryCard) and effect:
                score += effect.victory_points(self, player)

        return score

    def is_winner(self, player: int) -> bool:
        return self.get_player_score(player) == max(self.get_player_score(p) for p in self.players)

    def get_player_card_counts(self, player: int) -> Counter:
        p_state: PlayerState = self.player_states[player]

        return p_state.get_card_counts()

    def is_degenerate(self) -> bool:
        return self.supply[Curse] == 0 and self.supply[Copper] == 0 and any(p_state.is_degenerate() for p_state in self.player_states)

    def is_game_over(self) -> bool:
        return self.supply.is_game_over()

    def advance_phase(self):
        p_state: PlayerState = self.player_states[self.player]
        if self.phase == Phase.ActionPhase:
            logging.info('====ACTION PHASE====')
            if p_state.actions == 0 or p_state.get_action_card_count(Zone.Hand) == 0:
                self.phase = Phase.TreasurePhase
            else:
                self.decision.text = 'Choose an action to play'
                self.decision.select_cards(None, 0, 1)
                for card in p_state.hand:
                    if isinstance(card, ActionCard):
                        self.decision.add_unique_card(card)
        if self.phase == Phase.TreasurePhase:
            logging.info('====TREASURE PHASE====')
            if p_state.get_treasure_card_count(Zone.Hand) == 0:
                self.phase = Phase.BuyPhase
            else:
                self.decision.text = 'Choose a treasure to play'
                self.decision.select_cards(None, 0, 1)

                for card in p_state.hand:
                    if (isinstance(card, TreasureCard)):
                        self.decision.add_unique_card(card)
        if self.phase == Phase.BuyPhase and len(self.events) == 0:
            logging.info('====BUY PHASE====')
            if p_state.buys == 0:
                self.phase = Phase.CleanupPhase
            else:
                self.decision.text = 'Choose a card to buy'
                self.decision.select_cards(None, 0, 1)
                i = 0
                logging.info(f'Player {self.player} has ${p_state.coins}')
                # TODO: Refactor
                for cardClass, cardCount in self.supply.items():
                    card = cardClass()
                    if self.get_supply_cost(card) <= p_state.coins and cardCount > 0:
                        self.decision.card_choices.append(card)
                    i += 1

                if len(self.decision.card_choices) == 0:
                    self.decision.type = DecisionType.DecisionNone
                    self.phase = Phase.CleanupPhase
                    logging.info(f'Player {self.player} cannot afford to buy any cards')
        if self.phase == Phase.CleanupPhase:
            logging.debug('====CLEANUP PHASE====')
            self.update_play_area(self.player)
            self.discard_hand(self.player)
            self.draw_hand(self.player)

            logging.info(f'Player {self.player} ends turn {p_state.turns}')

            if self.is_game_over() or self.is_degenerate():
                self.decision.type = DecisionType.DecisionGameOver
                return

            self.player = (self.player + 1) % len(self.player_states)
            self.phase = Phase.ActionPhase
            p_state: PlayerState = self.player_states[self.player]
            p_state.actions = 1
            p_state.buys = 1
            p_state.coins = 0
            p_state._turns += 1

    def advance_next_decision(self):
        if self.decision.type == DecisionType.DecisionGameOver:
            return
        if self.decision.type != DecisionType.DecisionNone:
            if self.decision.is_trivial():
                self.process_decision(self.decision.trivial_response())
                self.advance_next_decision()
            return

        if len(self.events) == 0:
            self.advance_phase()
        else:
            # Inject a MoatReveal event in response to an attack card
            last_event = self.events[-1]
            # skipEventProcessing = False

            if last_event.is_attack():
                attacked_player = last_event.attacked_player()
                if attacked_player == -1:
                    raise ValueError('Invalid attacked player')
                p_state: PlayerState = self.player_states[last_event.attacked_player()]
                annotations = last_event.get_attack_annotations()

                if not annotations.moat_processed and p_state.contains_card(Moat(), Zone.Hand):
                    self.events.append(MoatReveal(Moat(), attacked_player))
                    last_event.annotations.moat_processed = True

            event_completed = self.events[-1].advance(self)
            if event_completed:
                self.events.pop()
                # skipEventProcessing = True

            # if not skipEventProcessing:
            #     eventCompleted = last_event.advance(self)
            #     if eventCompleted:
            #         destroy_next_event = last_event.destroy_next_event_on_stack()
            #         self.events.pop()
            #         del last_event
            #         if destroy_next_event:
            #             nextEvent = self.events.pop()
            #             del nextEvent

        if self.decision.type == DecisionType.DecisionNone:
            self.advance_next_decision()

        if self.decision.controlling_player == -1:
            self.decision.controlling_player = self.player

        if self.decision.type == DecisionType.DecisionSelectCards:
            self.decision.max_cards = min(self.decision.max_cards, len(self.decision.card_choices))

        if self.decision.is_trivial():
            self.process_decision(self.decision.trivial_response())
            self.advance_next_decision()

    def new_game(self):
        playerCount = len(self.player_states)
        for i in range(playerCount):
            self.draw_hand(i)

        # self.player = random.randint(0, playerCount-1)
        logging.info(f'Player {self.player} starts')


class AttackAnnotations:
    def __init__(self):
        self.moat_processed = False


class Event(ABC):
    @abstractmethod
    def advance(self, s: State) -> bool:
        pass

    def is_attack(self) -> bool:
        return False

    def attacked_player(self) -> int:
        return -1

    def get_attack_annotations(self) -> AttackAnnotations:
        return None

    def can_process_decisions(self) -> bool:
        return False

    def destroy_next_event_on_stack(self) -> bool:
        return False

    def process_decision(self, s: State, response: DecisionResponse):
        logging.warning('Event does not support decisions')

    def __repr__(self):
        return str(self)


class DrawCard(Event):
    def __init__(self, player: int) -> None:
        self.event_player = player

    def advance(self, s: State) -> bool:
        s.draw_card(self.event_player)
        return True

    def __str__(self):
        return 'Draw'


class DiscardCard(Event):
    def __init__(self, zone: DiscardZone, player: int, card: Card) -> None:
        self.zone = zone
        self.player = player
        self.card = card

    def advance(self, s: State):
        s.discard_card(self.player, self.card, self.zone)
        return True

    def __str__(self):
        return 'DiscardCard'


class GainCard(Event):
    def __init__(self, zone: GainZone, player: int, card: Card, bought=False, is_attack=False):
        self.player = player
        self.zone = zone
        self.card = card
        self.bought = bought
        self._is_attack = is_attack
        self.state = TriggerState.TriggerNone
        self.annotations = AttackAnnotations()

    def is_attack(self):
        return self._is_attack

    def attacked_player(self):
        return self.player

    def can_process_decisions(self):
        return True

    def get_attack_annotations(self):
        return self.annotations

    def process_decision(self, s: State, response: DecisionResponse):
        # TODO: Patch this when implementing Watchtower and Royal Seal
        self.state = TriggerState.TriggerProcessed

    def advance(self, s: State):
        s.gain_card(self.player, self.card, self.zone, self.bought)
        return True

    def __str__(self):
        return 'Gain'


class ReorderCards(Event):
    def __init__(self, cards: List[Card], player: int):
        self._cards = cards
        self._player = player

    def advance(self, s: State):
        p_state: PlayerState = s.player_states[self._player]
        _n = len(self._cards)
        if _n > 0:
            # TODO: Update when the order of deck matters to feature state
            p_state._deck[-_n:] = self._cards

        return True


class DiscardDownToN(Event):
    def __init__(self, card: Card, player: int, hand_size: int):
        self.card = card
        self.player = player
        self.hand_size = hand_size
        self.done = False
        self.annotations = AttackAnnotations()

    def is_attack(self) -> bool:
        return True

    def can_process_decisions(self) -> bool:
        return True

    def get_attack_annotations(self):
        return self.annotations

    def process_decision(self, s: State, response: DecisionResponse):
        for card in response.cards:
            s.events.append(DiscardCard(DiscardZone.DiscardFromHand, s.decision.controlling_player, card))
        self.done = True

    def attacked_player(self) -> int:
        return self.player

    def advance(self, s: State):
        if self.done:
            return True

        current_hand_size = len(s.player_states[self.player].hand)
        cards_to_discard = current_hand_size - self.hand_size

        if cards_to_discard <= 0:
            logging.info(f'Player {self.player} has cannot discard down: has {current_hand_size}')
            return True

        s.decision.select_cards(self.card, cards_to_discard, cards_to_discard)
        s.decision.card_choices = s.player_states[self.player].hand
        s.decision.controlling_player = self.player

        s.decision.text = f'Player {self.player}: choose {cards_to_discard} card(s) to discard:'
        return False

    def __str__(self):
        return f'DD{self.hand_size}'


class TrashCard(Event):
    def __init__(self, zone: Zone, player: int, card: Card) -> None:
        self.zone = zone
        self.player = player
        self.card = card

    def advance(self, s: State):
        s.trash_card(self.card, self.zone, self.player)
        return True

    def __str__(self):
        return 'Trash'


class RemodelExpand(Event):
    def __init__(self, source: Card, gained_value: int):
        self.source = source
        self.gained_value = gained_value
        self.trashed_card = None
        self.done = False

    def can_process_decisions(self):
        return True

    def process_decision(self, s: State, response: DecisionResponse):
        if not self.trashed_card:
            self.trashed_card = response.cards[0]
            s.events.append(TrashCard(Zone.Hand, s.player, self.trashed_card))
        else:
            s.events.append(GainCard(GainZone.GainToDiscard, s.player, response.cards[0]))
            self.done = True

    def advance(self, s: State):
        if self.done:
            return True

        s.decision.gain_card_from_supply(s, self.source, 0, self.trashed_card.get_coin_cost() + self.gained_value)
        if not s.decision.card_choices:
            s.decision.type = DecisionType.DecisionNone
            logging.info(f'Player {s.player} cannot gain any cards')
            return True
        return False

    def __str__(self):
        return 'RemodelExpand'


class EventArtisan(Event):
    def __init__(self, source: Card):
        self.source = source
        self.gained_card = None
        self.done = False

    def advance(self, s: State):
        if self.done:
            return True
        s.decision.gain_card_from_supply(s, self.source, 0, 5)
        s.decision.text = 'Choose a card to gain'
        return False

    def can_process_decisions(self):
        return True

    def process_decision(self, s: State, response: DecisionResponse):
        p_state: PlayerState = s.player_states[s.player]
        if not self.gained_card:
            self.gained_card = response.cards[0]
            s.events.append(PutOnDeckDownToN(self.source, s.player, len(p_state.hand)))
            s.events.append(GainCard(GainZone.GainToHand, s.player, response.cards[0]))
            self.done = True

    def __str__(self):
        return 'EventArtisan'


class EventMine(Event):
    def __init__(self, source: Card):
        self.source = source
        self.trashed_card = None
        self.done = False

    def can_process_decisions(self):
        return True

    def advance(self, s: State):
        if self.done:
            return True
        s.decision.gain_treasure_from_supply(s, self.source, 0, s.get_supply_cost(self.trashed_card) + 3)
        if not s.decision.card_choices:
            s.decision.type = DecisionType.DecisionNone
            logging.info(f'Player {s.player} Cannot gain any cards')
            return True
        return False

    def process_decision(self, s: State, response: DecisionResponse):
        if not self.trashed_card:
            self.trashed_card = response.cards[0]
            s.events.append(TrashCard(Zone.Hand, s.player, self.trashed_card))
        else:
            s.events.append(GainCard(GainZone.GainToHand, s.player, response.cards[0], False, False))
            self.done = True

    def __str__(self):
        return 'EventMine'


class EventLibrary(Event):
    def __init__(self, source: Card):
        self.source = source
        self.decision_card = None
        self.done_drawing = False
        self.library_zone = []

    # TODO: Fix this probably broken method
    def advance(self, s: State):
        p_state: PlayerState = s.player_states[s.player]
        current_hand_size = len(p_state.hand)
        if current_hand_size < 7 and not self.done_drawing:
            revealed_card = None
            if not p_state._deck:
                s.shuffle(s.player)
            else:
                return True
            if not p_state._deck:
                logging.info(f'Player {s.player} tries to draw, but has no cards left')
                self.done_drawing = True
            else:
                revealed_card = p_state._deck[-1]
                if isinstance(revealed_card, ActionCard):
                    self.decision_card = revealed_card
                    s.decision.make_discrete_choice(self.source, 2)
                    logging.info(f'Player {s.player} reveals {revealed_card}')
                    s.decision.text = f'Set aside {self.decision_card}? 0. Yes 1. No'
                else:
                    logging.info(f'Player {s.player} draws {revealed_card}')
                    s.move_card(s.player, revealed_card, Zone.Deck, Zone.Hand)
                return False
        self.done_drawing = True
        for card in self.library_zone:
            s.events.append(DiscardCard(DiscardZone.DiscardFromSideZone, s.player, card))
        return True

    def can_process_decisions(self):
        return True

    def process_decision(self, s: State, response: DecisionResponse):
        p_state: PlayerState = s.player_states[s.player]
        if response.choice == 0:
            remove_card(self.decision_card, p_state._deck)
            self.library_zone.append(self.decision_card)
            logging.info(f'Player {s.player} sets aside {self.decision_card}')
        else:
            s.move_card(s.player, self.decision_card, Zone.Deck, Zone.Hand)
            logging.info(f'Player {s.player} puts {self.decision_card} into their hand')

    def __str__(self):
        return 'EventLibrary'


class EventSentry(Event):
    def __init__(self, source: Card, choices: List[Card]):
        self.source = source
        self.trashed = False
        self.discarded = False
        self.done = False
        self.choices = choices

    def advance(self, s: State):
        if self.done:
            return True

        if not self.trashed:
            s.decision.select_cards(self.source, 0, len(self.choices))
            s.decision.text = 'Select cards to trash'
        elif not self.discarded:
            s.decision.select_cards(self.source, 0, len(self.choices))
            s.decision.text = 'Select cards to discard'
        elif len(self.choices) == 2:
            s.decision.text = 'Select cards to reorder'
            s.decision.select_cards(self.source, len(self.choices), len(self.choices))
        else:
            return True

        s.decision.card_choices = self.choices
        return False

    def can_process_decisions(self):
        return True

    def process_decision(self, s: State, response: DecisionResponse):
        if not self.trashed:
            for card in response.cards:
                s.events.append(TrashCard(Zone.Deck, s.player, card))
                remove_card(card, self.choices)
            self.trashed = True
        elif not self.discarded:
            for card in response.cards:
                s.events.append(DiscardCard(DiscardZone.DiscardFromDeck, s.player, card))
                remove_card(card, self.choices)
            self.discarded = True
        else:
            s.events.append(ReorderCards(response.cards, s.player))
            self.done = True

        if not self.choices:
            self.done = True

    def __str__(self):
        return 'EventSentry'


class BureaucratAttack(Event):
    def __init__(self, source: Card, player: int):
        self.source = source
        self.player = player
        self.annotations = AttackAnnotations()

    def is_attack(self):
        return True

    def attacked_player(self):
        return self.player

    def advance(self, s: State):
        p_state: PlayerState = s.player_states[self.player]
        if p_state.get_victory_card_count(Zone.Hand) == 0:
            logging.info(f'Player {self.player} reveals a hand with no victory cards')
        else:
            s.decision.select_cards(self.source, 1, 1)
            s.decision.controlling_player = self.player

            for card in p_state.hand:
                if isinstance(card, VictoryCard):
                    s.decision.add_unique_card(card)
            s.decision.text = 'Choose a victory card to put on top of your deck'
        return True

    def get_attack_annotations(self):
        return self.annotations

    def __str__(self):
        return 'BureaucratAttack'


class MoatReveal(Event):
    def __init__(self, source: Card, player: int):
        self.source = source
        self.player = player
        self.done = False
        self.revealed = False

    def can_process_decisions(self):
        return True

    def destroy_next_event_on_stack(self):
        return self.revealed

    def advance(self, s: State):
        if self.done:
            return True

        s.decision.make_discrete_choice(self.source, 2)
        s.decision.controlling_player = self.player
        s.decision.text = 'Reveal Moat? 0. Yes 1. No'
        return False

    def process_decision(self, s: State, response):
        if response.choice == 0:
            logging.info(f'Player {self.player} reveals Moat')
            # Pop MoatReveal and Corresponding attack card
            s.events = s.events[:-2]
        else:
            logging.info(f'Player {self.player} does not reveal Moat')
        self.revealed = True if response.choice == 0 else False
        self.done = True

    def __str__(self):
        return 'MoatReveal'


class PutOnDeckDownToN(Event):
    def __init__(self, source: Card, player: int, handSize: int, is_attack=False):
        self.source = source
        self.player = player
        self.handSize = handSize
        self.done = False
        self.annotations = AttackAnnotations if is_attack else None
        self._is_attack = is_attack

    def is_attack(self):
        return self._is_attack

    def attacked_player(self):
        return self.player if self._is_attack else -1

    def get_attack_annotations(self):
        return self.annotations

    def advance(self, s: State):
        if self.done:
            return True

        current_hand_size = len(s.player_states[self.player].hand)
        cards_to_discard = current_hand_size - self.handSize

        if cards_to_discard <= 0:
            logging.info(f'Player {self.player} has {current_hand_size} cards in hand')
            return True

        s.decision.select_cards(self.source, cards_to_discard, cards_to_discard)
        s.decision.card_choices = s.player_states[self.player].hand
        s.decision.controlling_player = self.player
        s.decision.text = f'Choose {cards_to_discard} card(s) to put on top of your deck:'
        return False

    def can_process_decisions(self):
        return True

    def process_decision(self, s: State, response: DecisionResponse):
        for card in response.cards:
            logging.info(f'Player {self.player} puts {card} on top of their deck')
            s.move_card(self.player, card, Zone.Hand, Zone.Deck)
        self.done = True

    def __str__(self):
        return f'PD{self.handSize}'


class PlayActionNTimes(Event):
    def __init__(self, source: Card, count: int):
        self.source = source
        self.target = None
        self.count = count

    def can_process_decisions(self):
        return True

    def advance(self, s: State):
        if self.count == 0:
            return True

        p_state: PlayerState = s.player_states[s.player]

        if not self.target:
            if p_state.get_action_card_count(Zone.Hand) == 0:
                logging.info(f'Player {s.player} has no actions to play')
                return True

            s.decision.select_cards(self.source, 1, 1)
            s.decision.text = "Select an action to play"

            for card in p_state.hand:
                if isinstance(card, ActionCard):
                    s.decision.add_unique_card(card)
        else:
            logging.info(f'Player {s.player} plays {self.target}')
            self.count -= 1
            s.process_action(self.target)
        return False

    def process_decision(self, s: State, response: DecisionResponse):
        p_state: PlayerState = s.player_states[s.player]
        self.target = response.cards[0]
        # TODO: Update feature after adding Throne Room
        target = remove_first_card(self.target, p_state.hand)

        target.copies = self.count
        p_state._play_area.append(target)

    def __str__(self):
        return f'Play{self.count}'
