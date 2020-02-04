from cardeffect import CardEffect
from actioncard import *
from cursecard import *
from enums import *
from state import *
from treasurecard import Copper, Silver, TreasureCard
from utils import containsCard

class CellarEffect(CardEffect):
    def __init__(self):
        self.c = Cellar()

    def playAction(self, s: State):
        if s.playerStates[s.player].hand:
            s.decision.selectCards(self.c, 0, len(s.playerStates[s.player].hand))
            s.decision.text = "Select cards to discard"
            s.decision.cardChoices = s.playerStates[s.player].hand
        else:
            print(f'Player {s.player} has no cards to discard')

    def canProcessDecisions(self):
        return True

    def processDecision(self, s: State, response: DecisionResponse):
        for _ in response.cards:
            s.events.append(DrawCard(s.player))
        for card in response.cards:
            s.events.append(DiscardCard(DiscardZone.DiscardFromHand, s.player, card))

class ChapelEffect(CardEffect):
    def __init__(self):
        self.c = Chapel()

    def playAction(self, s:State):
        if s.playerStates[s.player].hand:
            s.decision.selectCards(self.c, 0, 4)
            s.decision.text = "Select up to 4 cards to trash"
            s.decision.cardChoices = s.playerStates[s.player].hand
        else:
            print(f'Player {s.player} has no cards to trash')

    def canProcessDecisions(self):
        return True

    def processDecision(self, s: State, response: DecisionResponse):
        print(f'processing decision for ChapelEffect')
        for card in response.cards:
            s.events.append(TrashCard(Zone.Hand, s.player, card))
            print(f'Appended TrashCard to event stack')

class MoneylenderEffect(CardEffect):
    def __init__(self):
        self.c = Moneylender()

    def playAction(self, s: State):
        trashIdx = getFirstIndex(Copper(), s.playerStates[s.player].hand)
        if trashIdx >= 0:
            s.events.append(TrashCard(Zone.Hand, s.player, s.playerStates[s.player].hand[trashIdx]))
            s.playerStates[s.player].coins += 3
        else:
            print(f'Player {s.player} has no coppers to trash')

class WorkshopEffect(CardEffect):
    def __init__(self):
        self.c = Workshop()

    def playAction(self, s: State):
        s.decision.gainCardFromSupply(s, self.c, 0, 4)
        if not s.decision.cardChoices:
            s.decision.type = DecisionType.DecisionNone
            print(f'Player {s.player} has no cards to gain')

    def canProcessDecisions(self):
        return True

    def processDecision(self, s: State, response: DecisionResponse):
        # TODO: record buy in ledger
        s.events.append(GainCard(GainZone.GainToDiscard, s.player, response.cards[0]))

class MilitiaEffect(CardEffect):
    def __init__(self):
        self.c = Militia()

    def playAction(self, s: State):
        for player in s.data.players:
            if player != s.player:
                s.events.append(DiscardDownToN(self.c, player, 3))

class RemodelEffect(CardEffect):
    def __init__(self):
        self.c = Remodel()

    def playAction(self, s: State):
        if s.playerStates[s.player].hand:
            s.decision.selectCards(self.c, 1, 1)
            s.decision.text = 'Select a card to trash:'
            s.decision.cardChoices = s.playerStates[s.player].hand
            s.events.append(RemodelExpand(self.c, 2))
        else:
            print(f'Player {s.player} has no cards to trash')

class MineEffect(CardEffect):
    def __init__(self):
        self.c = Mine()

    def playAction(self, s: State):
        pState= s.playerStates[s.player]
        if pState.getTreasureCardCount(pState.hand) > 0:
            s.decision.selectCards(self.c, 1, 1)
            s.decision.text = 'Select a treasure to trash:'
            for card in pState.hand:
                if isinstance(card, TreasureCard):
                    s.decision.addUniqueCard(card)
            s.events.append(EventMine(self.c))
        else:
            print(f'Player {s.player} has no treasures to trash')

class LibraryEffect(CardEffect):
    def __init__(self):
        self.c = Library()

    def playAction(self, s: State):
        if len(s.playerStates[s.player].hand) < 7:
            s.events.append(EventLibrary(self.c))
        else:
            print(f'Player {s.player} already has 7 cards in hand')

class WitchEffect(CardEffect):
    def __init__(self):
        self.c = Witch()

    def playAction(self, s: State):
        for player in s.data.players:
            if player != s.player:
                s.events.append(GainCard(GainZone.GainToDiscard, player, Curse(), False, True))

class GardensEffect(CardEffect):
    def playAction(self, s):
        return
    def victoryPoints(self, s: State, player: int):
        return s.playerStates[player].getTotalCards() // 10


class BureaucratEffect(CardEffect):
    def __init__(self):
        self.c = Bureaucrat()

    def playAction(self, s: State):
        s.events.append(GainCard(GainZone.GainToDeckTop, s.player, Silver()))
        for player in s.data.players:
            if player != s.player:
                s.events.append(BureaucratAttack(self.c, player))

    def canProcessDecisions(self):
        return True

    def processDecision(self, s: State, response: DecisionResponse):
        c = response.cards[0]
        pState = s.playerStates[s.decision.controllingPlayer]
        moveCard(c, pState.hand, pState.deck)
        print(f'Player {s.decision.controllingPlayer} puts {c} from their hand onto their deck')

class ThroneRoomEffect(CardEffect):
    def __init__(self):
        self.c = ThroneRoom()

    def playAction(self, s: State):
        s.events.append(PlayActionNTimes(self.c, 2))