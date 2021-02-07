import logging
from typing import List

from actioncard import (ActionCard, Artisan, Bandit, Bureaucrat, Cellar,
                        Chapel, Harbinger, Library, Militia, Mine, Moneylender,
                        Poacher, Remodel, Sentry, ThroneRoom, Vassal, Witch,
                        Workshop)
from card import Card
from cardeffect import CardEffect
from cursecard import Curse
from enums import DecisionType, DiscardZone, GainZone, Zone
from playerstate import PlayerState
from state import (BureaucratAttack, DecisionResponse, DiscardCard,
                   DiscardDownToN, DrawCard, EventArtisan, EventLibrary,
                   EventMine, EventSentry, GainCard, PlayActionNTimes,
                   RemodelExpand, State, TrashCard)
from treasurecard import Copper, Gold, Silver, TreasureCard
from utils import get_first_index
from victorycard import Gardens


class ArtisanEffect(CardEffect):
    def __init__(self):
        self.c = Artisan()

    def play_action(self, s: State):
        s.events.append(EventArtisan(self.c))


class BanditEffect(CardEffect):
    def __init__(self):
        self.c = Bandit()

    # TODO: Allow player to choose which treasure to trash
    def play_action(self, s: State):
        for player in s.players:
            if player != s.player:
                p_state: PlayerState = s.player_states[player]

                stolen = p_state._deck[-2:]

                if not stolen:
                    return True

                trashable = list(filter(lambda x: isinstance(x, TreasureCard) and not isinstance(x, Copper), stolen))

                trashed = None if not trashable else min(trashable, key=lambda x: x.get_treasure())
                for card in stolen:
                    if card != trashed:
                        s.events.append(DiscardCard(DiscardZone.DiscardFromDeck, player, card))

                if trashed:
                    s.events.append(TrashCard(Zone.Deck, player, trashed))

        s.events.append(GainCard(GainZone.GainToDiscard, s.player, Gold()))


class BureaucratEffect(CardEffect):
    def __init__(self):
        self.c = Bureaucrat()

    def play_action(self, s: State):
        s.events.append(GainCard(GainZone.GainToDeckTop, s.player, Silver()))
        for player in s.players:
            if player != s.player:
                s.events.append(BureaucratAttack(self.c, player))

    def can_process_decisions(self):
        return True

    def process_decision(self, s: State, response: DecisionResponse):
        c = response.cards[0]
        s.move_card(s.decision.controlling_player, c, Zone.Hand, Zone.Deck)


class CellarEffect(CardEffect):
    def __init__(self):
        self.c = Cellar()

    def play_action(self, s: State):
        if s.player_states[s.player].hand:
            s.decision.select_cards(self.c, 0, len(s.player_states[s.player].hand))
            s.decision.text = "Select cards to discard"
            s.decision.card_choices = s.player_states[s.player].hand
        else:
            logging.info(f'Player {s.player} has no cards to discard')

    def can_process_decisions(self):
        return True

    def process_decision(self, s: State, response: DecisionResponse):
        for _ in response.cards:
            s.events.append(DrawCard(s.player))
        for card in response.cards:
            s.events.append(DiscardCard(DiscardZone.DiscardFromHand, s.player, card))


class ChapelEffect(CardEffect):
    def __init__(self):
        self.c = Chapel()

    def play_action(self, s: State):
        if s.player_states[s.player].hand:
            s.decision.select_cards(self.c, 0, 4)
            s.decision.text = "Select up to 4 cards to trash"
            s.decision.card_choices = s.player_states[s.player].hand
        else:
            logging.info(f'Player {s.player} has no cards to trash')

    def can_process_decisions(self):
        return True

    def process_decision(self, s: State, response: DecisionResponse):
        for card in response.cards:
            s.events.append(TrashCard(Zone.Hand, s.player, card))


class GardensEffect(CardEffect):
    def play_action(self, s):
        return

    def victory_points(self, s: State, player: int):
        return s.player_states[player].num_cards // 10


class HarbingerEffect(CardEffect):
    def __init__(self):
        self.c = Harbinger()

    def play_action(self, s: State):
        p_state: PlayerState = s.player_states[s.player]
        if len(p_state._discard) > 0:
            s.decision.select_cards(self.c, 0, 1)
            s.decision.card_choices = p_state._discard
            s.decision.text = 'Choose a card from discard to move'
        else:
            logging.info(f'Harbinger has no effect: player {s.player} has empty discard')

    def can_process_decisions(self):
        return True

    def process_decision(self, s: State, response: DecisionResponse):
        p_state: PlayerState = s.player_states[s.player]
        card = p_state._discard[response.choice]
        s.move_card(s.player, card, Zone.Discard, Zone.Deck)


class LibraryEffect(CardEffect):
    def __init__(self):
        self.c = Library()

    def play_action(self, s: State):
        if len(s.player_states[s.player].hand) < 7:
            s.events.append(EventLibrary(self.c))
        else:
            logging.info(f'Player {s.player} already has 7 cards in hand')


class MilitiaEffect(CardEffect):
    def __init__(self):
        self.c = Militia()

    def play_action(self, s: State):
        for player in s.players:
            if player != s.player:
                s.events.append(DiscardDownToN(self.c, player, 3))


class MineEffect(CardEffect):
    def __init__(self):
        self.c = Mine()

    def play_action(self, s: State):
        p_state: PlayerState = s.player_states[s.player]
        if s.get_treasure_card_count(s.player, Zone.Hand) > 0:
            s.decision.select_cards(self.c, 1, 1)
            s.decision.text = 'Select a treasure to trash:'
            for card in p_state.hand:
                if isinstance(card, TreasureCard):
                    s.decision.add_unique_card(card)
            s.events.append(EventMine(self.c))
        else:
            logging.info(f'Player {s.player} has no treasures to trash')


class MoneylenderEffect(CardEffect):
    def __init__(self):
        self.c = Moneylender()

    def play_action(self, s: State):
        trashIdx = get_first_index(Copper(), s.player_states[s.player].hand)
        if trashIdx >= 0:
            s.events.append(TrashCard(Zone.Hand, s.player, s.player_states[s.player].hand[trashIdx]))
            s.player_states[s.player].coins += 3
        else:
            logging.info(f'Player {s.player} has no coppers to trash')


class PoacherEffect(CardEffect):
    def __init__(self):
        self.c = Poacher()

    def play_action(self, s: State):
        p_state: PlayerState = s.player_states[s.player]
        num_empty_supply = s.supply.empty_stack_count
        if num_empty_supply > 0 and p_state.hand:
            numCards = min(len(p_state.hand), num_empty_supply)
            s.decision.select_cards(self.c, numCards, numCards)
            s.decision.card_choices = p_state.hand
            s.decision.text = 'Choose card(s) to discard'

    def can_process_decisions(self):
        return True

    def process_decision(self, s: State, response: DecisionResponse):
        for card in response.cards:
            s.events.append(DiscardCard(DiscardZone.DiscardFromHand, s.player, card))


class RemodelEffect(CardEffect):
    def __init__(self):
        self.c = Remodel()

    def play_action(self, s: State):
        if s.player_states[s.player].hand:
            s.decision.select_cards(self.c, 1, 1)
            s.decision.text = 'Select a card to trash:'
            s.decision.card_choices = s.player_states[s.player].hand
            s.events.append(RemodelExpand(self.c, 2))
        else:
            logging.info(f'Player {s.player} has no cards to trash')


class SentryEffect(CardEffect):
    def __init__(self):
        self.c = Sentry()

    def play_action(self, s: State):
        player: int = s.player
        p_state: PlayerState = s.player_states[player]
        zone_size = p_state.zone_size(Zone.Deck)

        if zone_size == 0:
            return

        n_choices = min(2, zone_size)
        card_choices = p_state._deck[-n_choices:]
        s.events.append(EventSentry(self.c, card_choices))


class ThroneRoomEffect(CardEffect):
    def __init__(self):
        self.c = ThroneRoom()

    def play_action(self, s: State):
        s.events.append(PlayActionNTimes(self.c, 2))


class VassalEffect(CardEffect):
    def __init__(self):
        self.c = Vassal()

    def play_action(self, s: State):
        p_state: PlayerState = s.player_states[s.player]
        deck: List[Card] = p_state._deck
        n_choices = 1 if (not deck or not isinstance(deck[-1], ActionCard)) else 2
        s.decision.make_discrete_choice(self.c, n_choices)
        s.decision.controlling_player = s.player
        s.decision.text = '0. Discard' if n_choices == 1 else f'0. Discard 1. Play {deck[-1]}'

    def can_process_decisions(self):
        return True

    def process_decision(self, s: State, response: DecisionResponse):
        p_state: PlayerState = s.player_states[s.player]
        deck: List[Card] = p_state._deck

        if not deck:
            return

        card: Card = deck[-1]
        if response.choice == 0:
            s.events.append(DiscardCard(DiscardZone.DiscardFromDeck, s.player, card))
        else:
            s.play_card(s.player, card, zone=Zone.Deck)
            s.process_action(card)


class WitchEffect(CardEffect):
    def __init__(self):
        self.c = Witch()

    def play_action(self, s: State):
        for player in s.players:
            if player != s.player:
                s.events.append(GainCard(GainZone.GainToDiscard, player, Curse(), False, True))


class WorkshopEffect(CardEffect):
    def __init__(self):
        self.c = Workshop()

    def play_action(self, s: State):
        s.decision.gain_card_from_supply(s, self.c, 0, 4)
        if not s.decision.card_choices:
            s.decision.type = DecisionType.DecisionNone
            logging.info(f'Player {s.player} has no cards to gain')

    def can_process_decisions(self):
        return True

    def process_decision(self, s: State, response: DecisionResponse):
        # TODO: record buy in ledger
        s.events.append(GainCard(GainZone.GainToDiscard, s.player, response.cards[0]))


# TODO: Implement Sentry, Bandit
BASE_EFFECT_MAP = {
    Artisan: ArtisanEffect,
    Bandit: BanditEffect,
    Bureaucrat: BureaucratEffect,
    Cellar: CellarEffect,
    Chapel: ChapelEffect,
    Gardens: GardensEffect,
    Harbinger: HarbingerEffect,
    Library: LibraryEffect,
    Militia: MilitiaEffect,
    Mine: MineEffect,
    Moneylender: MoneylenderEffect,
    Poacher: PoacherEffect,
    Remodel: RemodelEffect,
    Sentry: SentryEffect,
    ThroneRoom: ThroneRoomEffect,
    Vassal: VassalEffect,
    Witch: WitchEffect,
    Workshop: WorkshopEffect,
}


def get_card_effect(card: Card) -> CardEffect:
    for k, v in BASE_EFFECT_MAP.items():
        if isinstance(card, k):
            return v()
    return None
