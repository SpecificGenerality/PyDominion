import logging

from actioncard import *
from cardeffect import CardEffect
from cursecard import *
from enums import *
from state import *
from treasurecard import Copper, Silver, TreasureCard
from utils import contains_card, move_card

class ArtisanEffect(CardEffect):
    def __init__(self):
        self.c = Artisan()

    def play_action(self, s: State):
        s.events.append(EventArtisan(self.c))

class SentryEffect(CardEffect):
    def __init__(self):
        self.c = Sentry()

    def play_action(self, s: State):
        deck = s.player_states[s.player].deck
        if not deck:
            return

        numCards = min(2, len(deck))
        s.decision.select_cards(self.c, 0, numCards)
        s.decision.card_choices = deck[-numCards:]
        s.decision.text = 'Select cards to discard'

    def can_process_decisions(self):
        return True

    def process_decision(self, s: State, response: DecisionResponse):
        for card in response.cards:
            s.events.append(DiscardCard(DiscardZone.DiscardFromDeck, s.player, card))
        s.events.append(EventSentry(self.c, s.player, response.cards.copy()))


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

    def play_action(self, s:State):
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

class MilitiaEffect(CardEffect):
    def __init__(self):
        self.c = Militia()

    def play_action(self, s: State):
        for player in s.players:
            if player != s.player:
                s.events.append(DiscardDownToN(self.c, player, 3))

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

class MineEffect(CardEffect):
    def __init__(self):
        self.c = Mine()

    def play_action(self, s: State):
        pState = s.player_states[s.player]
        if pState.get_treasure_card_count(pState.hand) > 0:
            s.decision.select_cards(self.c, 1, 1)
            s.decision.text = 'Select a treasure to trash:'
            for card in pState.hand:
                if isinstance(card, TreasureCard):
                    s.decision.add_unique_card(card)
            s.events.append(EventMine(self.c))
        else:
            logging.info(f'Player {s.player} has no treasures to trash')

class LibraryEffect(CardEffect):
    def __init__(self):
        self.c = Library()

    def play_action(self, s: State):
        if len(s.player_states[s.player].hand) < 7:
            s.events.append(EventLibrary(self.c))
        else:
            logging.info(f'Player {s.player} already has 7 cards in hand')

class WitchEffect(CardEffect):
    def __init__(self):
        self.c = Witch()

    def play_action(self, s: State):
        for player in s.players:
            if player != s.player:
                s.events.append(GainCard(GainZone.GainToDiscard, player, Curse(), False, True))

class GardensEffect(CardEffect):
    def play_action(self, s):
        return
    def victory_points(self, s: State, player: int):
        return s.player_states[player].num_cards // 10

class PoacherEffect(CardEffect):
    def __init__(self):
        self.c = Poacher()

    def play_action(self, s: State):
        pState = s.player_states[s.player]
        numEmptySupply = s.numEmptySupply()
        if numEmptySupply > 0 and pState.hand:
            numCards = min(len(pState.hand), numEmptySupply)
            s.decision.select_cards(self.c, numCards, numCards)
            s.decision.card_choices = pState.hand
            s.decision.text = 'Choose card(s) to discard'

    def can_process_decisions(self):
        return True

    def process_decision(self, s: State, response: DecisionResponse):
        for card in response.cards:
            s.events.append(DiscardCard(DiscardZone.DiscardFromHand, s.player, card))

class HarbingerEffect(CardEffect):
    def __init__(self):
        self.c = Harbinger()

    def play_action(self, s: State):
        pState = s.player_states[s.player]
        if len(pState.discard) > 0:
            s.decision.select_cards(self.c, 0, 1)
            s.decision.card_choices = pState.discard
            s.decision.text = 'Choose a card from discard to move'
        else:
            logging.info(f'Harbinger has no effect: player {s.player} has empty discard')

    def can_process_decisions(self):
        return True

    def process_decision(self, s: State, response: DecisionResponse):
        pState = s.player_states[s.player]
        move_card(pState.discard[response.choice], pState.discard, pState.deck)

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
        pState = s.player_states[s.decision.controllingPlayer]
        move_card(c, pState.hand, pState.deck)

class ThroneRoomEffect(CardEffect):
    def __init__(self):
        self.c = ThroneRoom()

    def play_action(self, s: State):
        s.events.append(PlayActionNTimes(self.c, 2))

# TODO: Implement Sentry, Bandit
BASE_EFFECT_MAP = {
    Artisan: ArtisanEffect,
    Chapel: ChapelEffect,
    Cellar: CellarEffect,
    Harbinger: HarbingerEffect,
    Workshop: WorkshopEffect,
    Bureaucrat: BureaucratEffect,
    Militia: MilitiaEffect,
    Moneylender: MoneylenderEffect,
    Remodel: RemodelEffect,
    ThroneRoom: ThroneRoomEffect,
    Library: LibraryEffect,
    Mine: MineEffect,
    Poacher: PoacherEffect,
    Witch: WitchEffect,
    Gardens: GardensEffect
}

def get_card_effect(card: Card) -> CardEffect:
    for k, v in BASE_EFFECT_MAP.items():
        if isinstance(card, k):
            return v()
    return None
