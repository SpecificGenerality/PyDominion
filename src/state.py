from abc import ABC
from card import Card
from actioncard import *
from treasurecard import TreasureCard
from victorycard import *
from config import GameConfig
from enums import *
from gamedata import GameData
from playerstate import PlayerState
from typing import List, Dict
from stateutils import *
from utils import *
import random
from collections import Counter

class DecisionResponse():
    def __init__(self, cards: List[Card]):
        self.cards = cards
        self.choice = -1
        self.singleCard = None

    def __str__(self) -> str:
        return f'Choices: {self.cards}\nChoice: {self.choice}\nSingle Card: {self.singleCard}'

class DecisionState():
    def __init__(self):
        self.type = DecisionType.DecisionNone
        self.cardChoices = []
        self.activeCard = None
        self.minCards = None
        self.maxCards = None
        self.text = None
        self.controllingPlayer = None

    def isTrivial(self) -> bool:
        return (self.type == DecisionType.DecisionSelectCards \
            and len(self.cardChoices) == 1 and self.minCards == 1 \
            and self.maxCards == 1)

    def trivialResponse(self) -> DecisionResponse:
        return DecisionResponse([self.cardChoices[0]])

    def selectCards(self, card: Card, minCards: int, maxCards: int):
        self.activeCard = card
        self.type = DecisionType.DecisionSelectCards
        self.minCards = minCards
        self.maxCards = maxCards
        self.controllingPlayer = -1
        self.cardChoices = []

    def makeDiscreteChoice(self, card: Card, optionCount: int):
        self.activeCard = card
        self.type = DecisionType.DecisionDiscreteChoice
        self.minCards = optionCount
        self.maxCards = optionCount
        self.controllingPlayer = -1

    def addUniqueCard(self, card: Card):
        if card not in self.cardChoices:
            self.cardChoices.append(card)

    def addUniqueCards(self, cardList: List[Card]):
        for card in cardList:
            self.addUniqueCard(card)

    def gainCardFromSupply(self, s, card: Card, minCost: int, maxCost: int):
        self.selectCards(card, 1, 1)
        for k, v in s.data.supply.items():
            supplyCard = k()
            cost = s.getSupplyCost(supplyCard)
            if v > 0 and cost >= minCost and cost <= maxCost:
                self.addUniqueCard(supplyCard)

    def gainTreasureFromSupply(self, s, card: Card, minCost: int, maxCost: int):
        self.selectCards(card, 1, 1)
        self.text = 'Select a treasure to gain:'

        for k, v in s.data.supply.items():
            supplyCard = k()
            cost = s.getSupplyCost(supplyCard)
            if v > 0 and cost >= minCost and cost <= maxCost:
                self.addUniqueCard(supplyCard)

    def printCardChoices(self):
        for i, card in enumerate(self.cardChoices):
            print(f'{i}: {card}')

class State:
    def __init__(self, config: GameConfig, data: GameData):
        self.playerStates = [PlayerState(config) for i in range(config.numPlayers)]
        self.phase = Phase.ActionPhase
        self.decision = DecisionState()
        self.player = 0
        self.data = data
        self.events = []

    def drawCard(self, player: int) -> None:
        pState = self.playerStates[player]
        if not pState.deck:
            self.shuffle(player)

        if not pState.deck:
            print(f'Player {player} tries to draw but has no cards')
        else:
            pState.hand.append(pState.deck.pop())

    def drawHand(self, player: int):
        numCardsToDraw = 5
        for i in range(numCardsToDraw):
            self.drawCard(player)
        print(f'Player {player} draws a new hand.')

    def discardCard(self, player: int, card: Card):
        pState = self.playerStates[player]
        moveCard(card, pState.hand, pState.discard)

    def discardHand(self, player: int):
        pState= self.playerStates[player]
        pState.discard += pState.hand
        pState.hand = []
        print(f'Player {player} discards their hand')

    def updatePlayArea(self, player: int):
        pState = self.playerStates[player]
        newPlayArea = []
        for card in pState.playArea:
            if card.turns_left > 1:
                moveCard(card, pState.playArea, newPlayArea)
        pState.discard += pState.playArea
        pState.playArea = newPlayArea

    def trashCard(self, card: Card, zone: Zone, player: int) -> None:
        pState = self.playerStates[player]

        if zone == Zone.Hand:
            if pState.hand:
                trashed_card = removeCard(card, pState.hand)
                if trashed_card:
                    self.data.trash.append(trashed_card)
                    print(f'Player {player} trashes {card} from hand.')
                else:
                    print(f'Player {player} fails to trash {card} from hand: card does not exist.')
                    exit()
            else:
                print(f'Player {player} hand is empty, trashing nothing.')
        elif zone == Zone.Deck:
            if pState.deck:
                topCard = pState.deck.pop()
                print(f'Player trashes {topCard}')
            else:
                print(f'Player {player} deck is empty, trashing nothing')
        elif zone == Zone.Play:
            if pState.playArea:
                trashed_card = removeCard(pState.playArea, card)
                if trashed_card:
                    self.data.trash.append(trashed_card)
                    print(f'Player {player} trashes {card} from play.')
                else: print(f'Player {player} fails to trash {card} from play: card does not exist.')
        else:
            raise Exception(f'Player {player} attemped to trash card from un-recognized zone.')

    def playCard(self, player: int, card: Card) -> None:
        pState = self.playerStates[player]
        moveCard(card, pState.hand, pState.playArea)

    def processAction(self, card: Card):
        import constants
        pState = self.playerStates[self.player]
        pState.actions += card.getPlusActions()
        pState.buys += card.getPlusBuys()
        pState.coins += card.getPlusCoins()
        print(f'Actions: {pState.actions}\nBuys: {pState.buys}\nCoins: {pState.coins}')
        for i in range(card.getPlusCards()):
            self.drawCard(self.player)

        effect = constants.getCardEffect(card)
        if effect:
            effect.playAction(self)

    def processTreasure(self, card: Card):
        import constants

        assert isinstance(card, TreasureCard), 'Attemped to processTreasure a non-treasure card'
        pState = self.playerStates[self.player]
        treasureValue = card.getTreasure()
        print(f'Player {self.player} gets ${treasureValue}')
        pState.coins += treasureValue
        pState.buys += card.getPlusBuys()

        effect = constants.getCardEffect(card)
        if effect:
            effect.playAction(self)

    def processDecision(self, response: DecisionResponse):
        print(f'{response}')
        if self.decision.type == DecisionType.DecisionGameOver:
            return
        assert self.decision.type != DecisionType.DecisionNone, 'No decision active'

        singleCard = response.singleCard
        if self.decision.type == DecisionType.DecisionSelectCards and self.decision.maxCards <= 1:
            if len(response.cards) == 1:
                singleCard = response.cards[0]
                assert len(response.cards) <= 1, 'Invalid number of cards in response'
                assert self.decision.minCards == 0 or singleCard != None, 'No response chosen'
            else:
                # do some asserts here
                pass

        self.decision.type = DecisionType.DecisionNone
        p = self.playerStates[self.player]

        if not self.decision.activeCard:
            if self.phase == Phase.ActionPhase:
                if singleCard is None:
                    print(f'Player {self.player} chooses not to play an action')
                    self.phase = Phase.TreasurePhase
                else:
                    print(f'Playing {singleCard}')
                    self.playCard(self.player, singleCard)
                    p.actions -= 1
                    self.processAction(singleCard)
            elif self.phase == Phase.TreasurePhase:
                if singleCard is None:
                    print(f'Player {self.player} chooses not to play a treasure')
                    self.phase = Phase.BuyPhase
                else:
                    self.playCard(self.player, singleCard)
                    self.processTreasure(singleCard)
            elif self.phase == Phase.BuyPhase:
                if singleCard is None:
                    print(f'Player {self.player} chooses not to buy a card')
                    self.phase = Phase.CleanupPhase
                else:
                    self.events.append(GainCard(GainZone.GainToDiscard, self.player, singleCard, True, False))
        else:
            import constants
            activeCardEffect = constants.getCardEffect(self.decision.activeCard)
            if activeCardEffect and activeCardEffect.canProcessDecisions():
                activeCardEffect.processDecision(self, response)
            elif len(self.events) > 0 and self.events[-1].canProcessDecisions():
                self.events[-1].processDecision(self, response)
            else:
                print(f'Error: Decision cannot be processed')

        if self.decision.controllingPlayer == -1:
            self.decision.controllingPlayer = self.player
        self.decision.maxCards = min(self.decision.maxCards, len(self.decision.cardChoices))

    def shuffle(self, player: int) -> None:
        pState = self.playerStates[player]
        random.shuffle(pState.discard)
        pState.deck = pState.deck + pState.discard
        pState.discard = []

    def numEmptySupply(self) -> int:
        n = 0
        for k, v in self.data.supply.items():
            n += 1 if v == 0 else 0
        return n

    # TODO: peddler, bridge, quarry, and plunder affect this method
    def getSupplyCost(self, card: Card) -> int:
        return card.getCoinCost()

    def getPlayerScore(self, player: int) -> int:
        import constants
        pState = self.playerStates[player]
        score = 0
        allCards = pState.getAllCards()

        for card in allCards:
            points = card.getVictoryPoints()
            score += card.getVictoryPoints()
            effect = constants.getCardEffect(card)
            if isinstance(card, VictoryCard) and effect:
                score += effect.victoryPoints(self, player)

        return score

    def getCardCounts(self, player: int) -> Counter:
        pState = self.playerStates[player]

        allCards = pState.getAllCards()
        counter = Counter([str(card) for card in allCards])
        return counter

    def isGameOver(self) -> bool:
        supply = self.data.supply
        if Colony in supply and supply[Colony] == 0:
            print(f'Game over. Colonies ran out.')
            return True
        elif supply[Province] == 0:
            print(f'Game over. Provinces ran out.')
            return True
        elif self.numEmptySupply() >= 3:
            print(f'Game over. Three supply piles ran out.')
            return True
        return False

    def advancePhase(self):
        pState = self.playerStates[self.player]
        # print(f'Play: {pState.playArea}')
        print(f'Hand: {pState.hand}')
        # print(f'Discard: {pState.discard}')
        # print(f'Deck: {pState.deck}')
        if self.phase == Phase.ActionPhase:
            print(f'====ACTION PHASE====')
            if pState.actions == 0 or pState.getActionCardCount(pState.hand) == 0:
                print(f'Advancing to treasure phase...')
                self.phase = Phase.TreasurePhase
            else:
                self.decision.text = 'Choose an action to play:'
                self.decision.selectCards(None, 0, 1)
                for card in pState.hand:
                    if isinstance(card, ActionCard):
                        self.decision.addUniqueCard(card)
        if self.phase == Phase.TreasurePhase:
            print(f'====TREASURE PHASE====')
            if pState.getTreasureCardCount(pState.hand) == 0:
                print(f'Advancing to buy phase...')
                self.phase = Phase.BuyPhase
            else:
                self.decision.text = 'Choose a treasure to play'
                self.decision.selectCards(None, 0, 1)

                for card in pState.hand:
                    if (isinstance(card, TreasureCard)):
                        self.decision.addUniqueCard(card)
        if self.phase == Phase.BuyPhase and len(self.events) == 0:
            print(f'====BUY PHASE====')
            if pState.buys == 0:
                print(f'Advancing to cleanup phase...')
                self.phase = Phase.CleanupPhase
            else:
                self.decision.text = 'Choose a card to buy'
                self.decision.selectCards(None, 0, 1)
                i = 0
                print(f'Player {self.player} has ${pState.coins}')
                for cardClass, cardCount in self.data.supply.items():
                    card = cardClass()
                    if self.getSupplyCost(card) <= pState.coins and cardCount > 0:
                        self.decision.cardChoices.append(card)
                    i += 1

                if len(self.decision.cardChoices) == 0:
                    self.decision.type = DecisionType.DecisionNone
                    print(f'Advancing to cleanup phase...')
                    self.phase = Phase.CleanupPhase
                    print(f'Player {self.player} cannot afford to buy any cards')
        if self.phase == Phase.CleanupPhase:
            print(f'====CLEANUP PHASE====')
            print(f'Play: {pState.playArea}')
            print(f'Hand: {pState.hand}')
            print(f'Discard: {pState.discard}')
            print(f'Deck: {pState.deck}')
            self.updatePlayArea(self.player)
            self.discardHand(self.player)
            self.drawHand(self.player)

            print(f'Player {self.player} ends their {pState.turns}th turn')

            if self.isGameOver():
                self.decision.type = DecisionType.DecisionGameOver
                return

            ## If above doesn't work, then use this and modify isGameOver method
            # if self.decision.type == DecisionType.DecisionGameOver:
            #     return

            self.player = (self.player + 1) % len(self.playerStates)
            self.phase = Phase.ActionPhase
            pState = self.playerStates[self.player]
            pState.actions = 1
            pState.buys = 1
            pState.coins = 0
            pState.turns += 1

    def advanceNextDecision(self):
        if self.decision.type == DecisionType.DecisionGameOver:
            return
        if self.decision.type != DecisionType.DecisionNone:
            if self.decision.isTrivial():
                self.processDecision(self.decision.trivialResponse())
                self.advanceNextDecision()
            return

        if len(self.events) == 0:
            print(f'Advancing phase...')
            self.advancePhase()
        else:
            currEvent = self.events[-1]
            print(f'{currEvent}')
            skipEventProcessing = False

            if currEvent.isAttack():
                attackedPlayer = currEvent.attackedPlayer()
                assert attackedPlayer != -1, 'Invalid Player'
                pState = self.playerStates[currEvent.attackedPlayer()]
                annotations = currEvent.getAttackAnnotations()

                if not annotations.moatProcessed and containsCard(Moat(), pState.hand):
                    self.events.append(MoatReveal(Moat(), attackedPlayer))
                    currEvent.annotations.moatProcessed = True
                    skipEventProcessing = True
            if not skipEventProcessing:
                eventCompleted = currEvent.advance(self)
                if eventCompleted:
                    destroyNextEvent = currEvent.destroyNextEventOnStack()
                    self.events.pop()
                    del currEvent
                    if destroyNextEvent:
                        nextEvent = self.events.pop()
                        del nextEvent

        if self.decision.type == DecisionType.DecisionNone:
            print(f'No decision needed...advancing to next decision')
            self.advanceNextDecision()

        if self.decision.controllingPlayer == -1:
            self.decision.controllingPlayer = self.player

        self.decision.maxCards = min(self.decision.maxCards, len(self.decision.cardChoices))
        if self.decision.isTrivial():
            self.processDecision(self.decision.trivialResponse())
            self.advanceNextDecision()

    def newGame(self):
        playerCount = len(self.playerStates)
        for i in range(playerCount):
            self.drawHand(i)

        self.player = random.randint(0, playerCount-1)
        print(f'Player {self.player} starts')
        self.playerStates[self.player].turns = 1

        self.advanceNextDecision()

class AttackAnnotations():
    def __init__(self):
        self.moatProcessed = False

class Event(ABC):
    @abstractmethod
    def advance(self, s: State) -> bool:
        pass

    def isAttack(self) -> bool:
        return False

    def attackedPlayer(self) -> int:
        return -1

    def getAttackAnnotations(self) -> AttackAnnotations:
        return None

    def canProcessDecisions(self) -> bool:
        return False

    def destroyNextEventOnStack(self) -> bool:
        return False

    def processDecision(self, s: State,  response: DecisionResponse):
        print(f'Event does not support decisions')

    def __repr__(self):
        return str(self)

class DrawCard(Event):
    def __init__(self, player: int) -> None:
        self.eventPlayer = player

    def advance(self, s: State) -> bool:
        s.drawCard(self.eventPlayer)
        return True

    def __str__(self):
        return f'Draw'

class DiscardCard(Event):
    def __init__(self, zone: DiscardZone, player: int, card: Card) -> None:
        self.zone = zone
        self.player = player
        self.card = card

    def advance(self, s: State):
        if self.zone == DiscardZone.DiscardFromHand:
            s.discardCard(self.player, self.card)
        else:
            print(f'DEBUG: Attempted to discard from non-hand zone')
        return True

    def __str__(self):
        return f'Discard.{id(self)}'

class GainCard(Event):
    def __init__(self, zone: GainZone, player: int, card: Card, bought=False, isAttack=False):
        self.player = player
        self.zone = zone
        self.card = card
        self.bought = bought
        self.is_attack = isAttack
        self.state = TriggerState.TriggerNone
        self.annotations = AttackAnnotations()

    def isAttack(self):
        return self.is_attack

    def attackedPlayer(self):
        return self.player

    def canProcessDecisions(self):
        return True

    def getAttackAnnotations(self):
        return self.annotations

    def processDecision(self, s: State, response: DecisionResponse):
        # TODO: Patch this when implementing Watchtower and Royal Seal
        self.state = TriggerState.TriggerProcessed

    def advance(self, s: State):
        pState = s.playerStates[self.player]
        supply = s.data.supply
        if supply[type(self.card)] > 0:
            if self.zone == GainZone.GainToHand:
                pState.hand.append(self.card)
                print(f'Player {self.player} gains {self.card} to hand')
            elif self.zone == GainZone.GainToDiscard:
                pState.discard.append(self.card)
                print(f'Player {self.player} gains {self.card} to discard')
            elif self.zone == GainZone.GainToDeckTop:
                pState.deck.append(self.card)
                print(f'Player {self.player} gains {self.card} to deck')

            supply[type(self.card)] -= 1

            if self.bought:
                cost = self.card.getCoinCost()
                print(f'Player {self.player} spends {cost} and buys {self.card}')
                pState.coins -= cost
                pState.buys -= 1
        else:
            if self.bought:
                print(f'Player {self.player} cannot buy {self.card}')
                pState.buys -= 1
            else:
                print(f'Player {self.player} cannot gain {self.card}')
        return True

    def __str__(self):
        return f'Gain'

class DiscardDownToN(Event):
    def __init__(self, card: Card, player: int, hand_size: int):
        self.card = card
        self.player = player
        self.hand_size = hand_size
        self.done = False
        self.annotations = AttackAnnotations()

    def isAttack(self) -> bool:
        return True

    def canProcessDecisions(self) -> bool:
        return True

    def getAttackAnnotations(self):
        return self.annotations

    def processDecision(self, s: State, response: DecisionResponse):
        for card in response.cards:
            s.events.append(DiscardCard(DiscardZone.DiscardFromHand, s.decision.controllingPlayer, card))
        self.done = True

    def attackedPlayer(self) -> int:
        return self.player

    def advance(self, s: State):
        if self.done:
            return True

        current_hand_size = len(s.playerStates[self.player].hand)
        cards_to_discard = current_hand_size - self.hand_size

        if cards_to_discard <= 0:
            print(f'Player {self.player} has cannot discard down: has {current_hand_size}')
            return True

        s.decision.selectCards(self.card, cards_to_discard, cards_to_discard)
        s.decision.cardChoices = s.playerStates[self.player].hand
        s.decision.controllingPlayer = self.player

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
        s.trashCard(self.card, self.zone, self.player)
        return True

    def __str__(self):
        return f'Trash'

class RemodelExpand(Event):
    def __init__(self, source: Card, gained_value: int):
        self.source = source
        self.gained_value = gained_value
        self.trashed_card = None
        self.done = False

    def canProcessDecisions(self):
        return True

    def processDecision(self, s: State, response: DecisionResponse):
        pState = s.playerStates[s.player]
        if not self.trashed_card:
            self.trashed_card = response.cards[0]
            s.events.append(TrashCard(Zone.Hand, s.player, self.trashed_card))
        else:
            s.events.append(GainCard(GainZone.GainToDiscard, s.player, response.cards[0]))
            self.done = True

    def advance(self, s: State):
        if self.done:
            return True

        s.decision.gainCardFromSupply(s, self.source, 0, self.trashed_card.getCoinCost() + self.gained_value)
        if not s.decision.cardChoices:
            s.decision.type = DecisionType.DecisionNone
            print(f'Player {s.player} cannot gain any cards')
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
        s.decision.gainCardFromSupply(s, self.source, 0, 5)
        s.decision.text = 'Choose a card to gain'
        return False

    def canProcessDecisions(self):
        return True

    def processDecision(self, s: State, response: DecisionResponse):
        pState = s.playerStates[s.player]
        if not self.gained_card:
            self.gained_card = response.cards[0]
            s.events.append(PutOnDeckDownToN(self.source, s.player, len(pState.hand)))
            s.events.append(GainCard(GainZone.GainToHand, s.player, response.cards[0]))
            self.done = True

    def __str__(self):
        return 'EventArtisan'

class EventMine(Event):
    def __init__(self, source: Card):
        self.source = source
        self.trashed_card = None
        self.done = False

    def canProcessDecisions(self):
        return True

    def advance(self, s: State):
        if self.done:
            return True
        s.decision.gainTreasureFromSupply(s, self.source, 0, s.getSupplyCost(self.trashed_card) + 3)
        if not s.decision.cardChoices:
            s.decision.type = DecisionType.DecisionNone
            print(f'Cannot gain any cards')
            return True
        return False

    def processDecision(self, s: State, response: DecisionResponse):
        pState = s.playerStates[s.player]
        if not self.trashed_card:
            self.trashed_card = response.cards[0]
            s.events.append(TrashCard(Zone.Hand, s.player, self.trashed_card))
        else:
            s.events.append(GainCard(GainZone.GainToHand, s.player, response.cards[0], False, False))
            self.done = True

    def __str__(self):
        return f'Mine'

class EventLibrary(Event):
    def __init__(self, source: Card):
        self.source = source
        self.decision_card = None
        self.done_drawing = False
        self.library_zone = []

    def canProcessDecisions(self):
        return True

    def processDecision(self, s: State, response: DecisionResponse):
        pState = s.playerStates[s.player]
        if response.choice == 0:
            self.library_zone.append(self.decision_card)
            print(f'Player {s.player} sets aside {self.decision_card}')
        else:
            pState.hand.append(self.decision_card)
            print(f'Player {s.player} puts {self.decision_card} into their hand')

    def advance(self, s: State):
        pState = s.playerStates[s.player]
        current_hand_size = len(pState.hand)
        if current_hand_size < 7 and self.done_drawing:
            revealed_card = None
            if not pState.deck:
                s.shuffle(s.player)
            if not pState.deck:
                print(f'Player {s.player} tries to draw, but has no cards left')
                self.done_drawing = True
            else:
                revealed_card = pState.deck.pop()
                if isinstance(revealed_card, ActionCard):
                    self.decision_card = revealed_card
                    s.decision.makeDiscreteChoice(self.source, 2)
                    print(f'Player {s.player} reveals {revealed_card}')
                    print(f'Set aside {self.decision_card}? Yes|No')
                else:
                    print(f'Player {s.player} draws {revealed_card}')
                    pState.hand.append(revealed_card)
                return False

        for card in self.library_zone:
            s.events.append(DiscardCard(DiscardZone.DiscardFromSideZone, s.player, card))
        return True

    def __str__(self):
        return f'Library'

class BureaucratAttack(Event):
    def __init__(self, source: Card, player: int):
        self.source = source
        self.player = player
        self.annotations = AttackAnnotations()

    def isAttack(self):
        return True

    def attackedPlayer(self):
        return self.player

    def advance(self, s: State):
        pState = s.playerStates[self.player]
        if pState.getVictoryCardCount(pState.hand) == 0:
            print(f'Player {self.player} reveals a hand with no victory cards')
        else:
            s.decision.selectCards(self.source, 1, 1)
            s.decision.controllingPlayer = self.player

            for card in pState.hand:
                if isinstance(card, VictoryCard):
                    s.decision.addUniqueCard(card)
            print(f'Choose a victory card to put on top of your deck:')
        return True

    def getAttackAnnotations(self):
        return self.annotations

    def __str__(self):
        return f'Bureaucrat'

class MoatReveal(Event):
    def __init__(self, source: Card, player: int):
        self.source = source
        self.player = player
        self.done = False
        self.revealed = False

    def canProcessDecisions(self):
        return True

    def destroyNextEventOnStack(self):
        return True

    def advance(self, s: State):
        if self.done:
            return True

        s.decision.makeDiscreteChoice(self.source, 2)
        s.decision.controllingPlayer = self.player
        s.decision.text = 'Reveal Moat?|Yes|No'
        return True

    def processDecision(self, s, response):
        if response.choice == 0:
            print(f'Player {self.player} reveals Moat')
        else:
            print(f'Player {self.player} does not reveal Moat')
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
        self.is_attack = is_attack

    def isAttack(self):
        return self.is_attack

    def attackedPlayer(self):
        return self.player if self.is_attack else -1

    def getAttackAnnotations(self):
        return self.annotations

    def advance(self, s: State):
        if self.done:
            return True

        currentHandSize = len(s.playerStates[self.player].hand)
        cardsToDiscard = currentHandSize - self.handSize

        if cardsToDiscard <= 0:
            print(f'Player {self.player} has {currentHandSize} cards in hand')
            return True

        s.decision.selectCards(self.source, cardsToDiscard, cardsToDiscard)
        s.decision.cardChoices = s.playerStates[self.player].hand
        s.decision.controllingPlayer = self.player
        s.decision.text = f'Choose {cardsToDiscard} card(s) to put on top of your deck:'
        return False

    def canProcessDecisions(self):
        return True

    def processDecision(self, s: State, response: DecisionResponse):
        pState = s.playerStates[self.player]
        for card in response.cards:
            print(f'Player {self.player} puts {card} on top of their deck')
            moveCard(card, pState.hand, pState.deck)
        self.done = True

    def __str__(self):
        return f'PD{self.handSize}'

class PlayActionNTimes(Event):
    def __init__(self, source: Card, count: int):
        self.source = source
        self.target = None
        self.count = count

    def canProcessDecisions(self):
        return True

    def advance(self, s: State):
        if self.count == 0:
            return True

        pState = s.playerStates[s.player]

        if not self.target:
            if pState.getActionCardCount(pState.hand) == 0:
                print(f'Player {s.player} has no actions to play')
                return True

            s.decision.selectCards(self.source, 1, 1)
            s.decision.text = "Select an action to play"

            for card in pState.hand:
                if isinstance(card, ActionCard):
                    s.decision.addUniqueCard(card)
        else:
            print(f'Player {s.player} plays {self.target}')
            self.count -= 1
            s.processAction(self.target)
        return False

    def processDecision(self, s: State, response: DecisionResponse):
        pState = s.playerStates[s.player]
        self.target = response.cards[0]
        target = removeCard(self.target, pState.hand)

        target.copies = self.count
        pState.playArea.append(target)

    def __str__(self):
        return f'Play{self.count}'