import logging
import random
from abc import ABC, abstractmethod
from collections import Counter
from typing import Dict, List

from actioncard import ActionCard, Merchant, Moat
from card import Card
from config import GameConfig
from cursecard import *
from enums import (DecisionType, DiscardZone, GainZone, Phase, TriggerState,
                   Zone)
from playerstate import PlayerState
from supply import Supply
from treasurecard import Copper, Silver, TreasureCard
from utils import (contains_card, get_first_index, move_card, remove_card,
                   remove_first_card)
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
        return (self.type == DecisionType.DecisionSelectCards \
            and len(self.card_choices) == 1 and self.min_cards == 1 \
            and self.max_cards == 1) or (self.type == DecisionType.DecisionDiscreteChoice and self.min_cards == self.max_cards == 1)

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

    def draw_card(self, player: int) -> None:
        p_state: PlayerState = self.player_states[player]
        if not p_state._deck:
            self.shuffle(player)

        if not p_state._deck:
            logging.info(f'Player {player} tries to draw but has no cards')
        else:
            p_state.hand.append(p_state._deck.pop())

    def draw_hand(self, player: int):
        num_cards_to_draw = 5
        for _ in range(num_cards_to_draw):
            self.draw_card(player)
        logging.info(f'Player {player} draws a new hand.')

    def discard_card(self, player: int, card: Card, cards: List[Card]):
        p_state: PlayerState = self.player_states[player]
        move_card(card, cards, p_state._discard)

    def discard_hand(self, player: int):
        p_state: PlayerState = self.player_states[player]
        p_state._discard += p_state.hand
        p_state.hand = []
        logging.info(f'Player {player} discards their hand')

    def update_play_area(self, player: int):
        p_state: PlayerState = self.player_states[player]
        new_play_area = []
        for card in p_state._play_area:
            if card.turns_left > 1:
                move_card(card, p_state._play_area, new_play_area)
        p_state._discard += p_state._play_area
        p_state._play_area = new_play_area

    def trash_card(self, card: Card, zone: Zone, player: int) -> None:
        p_state: PlayerState = self.player_states[player]
        # print(f'Trashing {card}')
        if zone == Zone.Hand:
            if p_state.hand:
                trashed_card = remove_card(card, p_state.hand)
                if trashed_card:
                    self.trash.append(trashed_card)
                    logging.info(f'Player {player} trashes {card} from hand.')
                else:
                    logging.error(f'Player {player} fails to trash {card} from hand: card does not exist.')
                    exit()
            else:
                logging.info(f'Player {player} hand is empty, trashing nothing.')
        elif zone == Zone.Deck:
            if p_state._deck:
                trashed_card = remove_card(card, p_state._deck)
                if trashed_card:
                    self.trash.append(trashed_card)
                    logging.info(f'Player trashes {trashed_card}')
            else:
                logging.info(f'Player {player} deck is empty, trashing nothing')
        elif zone == Zone.Play:
            if p_state._play_area:
                trashed_card = remove_card(p_state._play_area, card)
                if trashed_card:
                    self.trash.append(trashed_card)
                    logging.info(f'Player {player} trashes {card} from play.')
                else:
                    logging.error(f'Player {player} fails to trash {card} from play: card does not exist.')
        else:
            logging.error(f'Player {player} attemped to trash card from un-recognized zone.')

    def play_card(self, player: int, card: Card, zone: Zone = Zone.Hand) -> None:
        p_state: PlayerState = self.player_states[player]
        p_state.play_card(card, zone)

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

    def shuffle(self, player: int) -> None:
        p_state: PlayerState = self.player_states[player]
        p_state.shuffle()

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
            logging.info(f'====ACTION PHASE====')
            if p_state.actions == 0 or p_state.get_action_card_count(Zone.Hand) == 0:
                self.phase = Phase.TreasurePhase
            else:
                self.decision.text = 'Choose an action to play'
                self.decision.select_cards(None, 0, 1)
                for card in p_state.hand:
                    if isinstance(card, ActionCard):
                        self.decision.add_unique_card(card)
        if self.phase == Phase.TreasurePhase:
            logging.info(f'====TREASURE PHASE====')
            if p_state.get_treasure_card_count(Zone.Hand) == 0:
                self.phase = Phase.BuyPhase
            else:
                self.decision.text = 'Choose a treasure to play'
                self.decision.select_cards(None, 0, 1)

                for card in p_state.hand:
                    if (isinstance(card, TreasureCard)):
                        self.decision.add_unique_card(card)
        if self.phase == Phase.BuyPhase and len(self.events) == 0:
            logging.info(f'====BUY PHASE====')
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
            logging.debug(f'====CLEANUP PHASE====')
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

class AttackAnnotations():
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

    def process_decision(self, s: State,  response: DecisionResponse):
        logging.warning(f'Event does not support decisions')

    def __repr__(self):
        return str(self)

class DrawCard(Event):
    def __init__(self, player: int) -> None:
        self.event_player = player

    def advance(self, s: State) -> bool:
        s.draw_card(self.event_player)
        return True

    def __str__(self):
        return f'Draw'

class DiscardCard(Event):
    def __init__(self, zone: DiscardZone, player: int, card: Card) -> None:
        self.zone = zone
        self.player = player
        self.card = card

    def advance(self, s: State):
        p_state: PlayerState = s.player_states[self.player]
        if self.zone == DiscardZone.DiscardFromHand:
            s.discard_card(self.player, self.card, p_state.hand)
        elif self.zone == DiscardZone.DiscardFromDeck:
            s.discard_card(self.player, self.card, p_state._deck)
        elif self.zone == DiscardZone.DiscardFromSideZone:
            p_state._discard.append(self.card)
        else:
            logging.warning(f'Attempted to discard from non-hand zone')
        return True

    def __str__(self):
        return f'DiscardCard'

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
        p_state: PlayerState = s.player_states[self.player]
        supply: Supply = s.supply
        # TODO: Refactor

        if supply[type(self.card)] > 0:
            if self.zone == GainZone.GainToHand:
                p_state.hand.append(self.card)
                logging.info(f'Player {self.player} gains {self.card} to hand')
            elif self.zone == GainZone.GainToDiscard:
                p_state._discard.append(self.card)
                logging.info(f'Player {self.player} gains {self.card} to discard')
            elif self.zone == GainZone.GainToDeckTop:
                p_state._deck.append(self.card)
                logging.info(f'Player {self.player} gains {self.card} to deck')

            # TODO: Refactor
            supply[type(self.card)] -= 1

            if self.bought:
                cost = s.get_supply_cost(self.card)
                logging.info(f'Player {self.player} spends {cost} and buys {self.card}')
                p_state.coins -= cost
                p_state.buys -= 1
        else:
            if self.bought:
                logging.info(f'Player {self.player} cannot buy {self.card}')
                p_state.buys -= 1
            else:
                logging.info(f'Player {self.player} cannot gain {self.card}')
        return True

    def __str__(self):
        return f'Gain'

class ReorderCards(Event):
    def __init__(self, cards: List[Card], player: int):
        self._cards = cards
        self._player = player

    def advance(self, s: State):
        p_state: PlayerState = s.player_states[self._player]
        _n = len(self._cards)
        if _n > 0:
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
        return f'Trash'

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
        return f'RemodelExpand'

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
        return f'EventMine'

class EventLibrary(Event):
    def __init__(self, source: Card):
        self.source = source
        self.decision_card = None
        self.done_drawing = False
        self.library_zone = []

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
                revealed_card = p_state._deck.pop()
                if isinstance(revealed_card, ActionCard):
                    self.decision_card = revealed_card
                    s.decision.make_discrete_choice(self.source, 2)
                    logging.info(f'Player {s.player} reveals {revealed_card}')
                    s.decision.text = f'Set aside {self.decision_card}? 0. Yes 1. No'
                else:
                    logging.info(f'Player {s.player} draws {revealed_card}')
                    p_state.hand.append(revealed_card)
                return False
        self.done_drawing = True
        for card in self.library_zone:
            print(card)
            s.events.append(DiscardCard(DiscardZone.DiscardFromSideZone, s.player, card))
        return True

    def can_process_decisions(self):
        return True

    def process_decision(self, s: State, response: DecisionResponse):
        p_state: PlayerState = s.player_states[s.player]
        if response.choice == 0:
            self.library_zone.append(self.decision_card)
            logging.info(f'Player {s.player} sets aside {self.decision_card}')
        else:
            p_state.hand.append(self.decision_card)
            logging.info(f'Player {s.player} puts {self.decision_card} into their hand')

    def __str__(self):
        return f'EventLibrary'

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
            s.decision.text = f'Choose a victory card to put on top of your deck'
        return True

    def get_attack_annotations(self):
        return self.annotations

    def __str__(self):
        return f'BureaucratAttack'

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
        return f'MoatReveal'

class PutOnDeckDownToN(Event):
    def __init__(self, source: Card, player: int, handSize: int, is_attack = False):
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
        p_state: PlayerState = s.player_states[self.player]
        for card in response.cards:
            logging.info(f'Player {self.player} puts {card} on top of their deck')
            move_card(card, p_state.hand, p_state._deck)
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
        target = remove_first_card(self.target, p_state.hand)

        target.copies = self.count
        p_state._play_area.append(target)

    def __str__(self):
        return f'Play{self.count}'
