import logging
import random
from abc import ABC, abstractmethod
from typing import List

import numpy as np
import numpy.random
import torch
from sklearn.linear_model import LogisticRegression

from aiutils import load, softmax
from buyagenda import (BigMoneyBuyAgenda, BuyAgenda, DoubleWitchBuyAgenda,
                       RandomBuyAgenda, TDBigMoneyBuyAgenda,
                       TDEBigMoneyBuyAgenda)
from card import Card
from constants import BUY
from cursecard import Curse
from enums import DecisionType, Phase
from heuristics import PlayerHeuristic
from heuristicsutils import heuristic_select_cards
from mcts import GameTree
from mlp import PredictorMLP
from rollout import RandomRollout, RolloutModel, init_rollouts, load_rollout
from state import (DecisionResponse, DecisionState, DiscardDownToN, MoatReveal,
                   PutOnDeckDownToN, RemodelExpand, State)
from treasurecard import Copper
from utils import remove_first_card
from victorycard import Estate, VictoryCard


# feature decks as counts of each card, least squares regress each against scores + offset
# try random + greedy
# TODO: add reset function
class Player(ABC):
    @classmethod
    @abstractmethod
    def load(cls, **kwargs):
        pass

    @abstractmethod
    def makeDecision(self, s: State, response: DecisionResponse):
        '''Given the current state s of the game, make a decision given the choices in s and modify response.'''
        pass


class GreedyLogisticPlayer(Player):
    def __init__(self, model: LogisticRegression, train=False, tau=0.01):
        self.model: LogisticRegression = model
        self.train: bool = train
        self.tau = tau

    @classmethod
    def load(cls, **kwargs):
        if 'path' not in kwargs:
            raise KeyError('Model path missing from kwargs.')

        model = load(kwargs['path'])
        return cls(model)

    def makeDecision(self, s: State, response: DecisionResponse):
        d: DecisionState = s.decision
        p: int = s.player
        if s.phase == Phase.ActionPhase:
            assert False, 'GreedyPlayer does not support action cards yet'
        elif s.phase == Phase.TreasurePhase:
            response.single_card = d.card_choices[0]
        else:
            choices = d.card_choices + [None]

            X = s.lookahead_batch_featurize(choices).cpu()

            label_idx = np.argmin(self.model.classes_) if p == 1 else np.argmax(self.model.classes_)

            y = self.model.predict_proba(X)

            if self.train:
                card = np.random.choice(choices, p=softmax(y[:, label_idx], t=self.tau))
            else:
                card = choices[np.argmax(y[:, label_idx])]

            response.single_card = card


class GreedyMLPPlayer(Player):
    def __init__(self, model):
        self.model = model

    @classmethod
    def load(cls, **kwargs):
        if 'path' not in kwargs:
            raise KeyError('Model path missing from kwargs.')

        model = torch.load(kwargs['path'])
        model.cuda()
        return cls(model)

    def makeDecision(self, s: State, response: DecisionResponse):
        d: DecisionState = s.decision
        p: int = s.player
        if s.phase == Phase.ActionPhase:
            assert False, 'GreedyMLPPlayer does not support action cards yet'
        elif s.phase == Phase.TreasurePhase:
            response.single_card = d.card_choices[0]
        else:
            choices = d.card_choices + [None]

            X = s.lookahead_batch_featurize(choices)

            label_idx = 0 if p == 1 else 2

            y_pred = self.model.forward(X)

            card_idx = torch.argmax(y_pred[:, label_idx])

            response.single_card = choices[card_idx]


class PredictorMLPPlayer(Player):
    def __init__(self, model: PredictorMLP, train: bool = True, tau=0.5):
        self.model = model
        self.train = train
        self.tau = tau
        self.iters = 0
        self.min_eps = 0.1

    @classmethod
    def load(cls, **kwargs):
        if 'path' not in kwargs:
            raise KeyError('Model path missing from kwargs.')

        model = load(kwargs.pop('path'), **kwargs)
        return cls(model, train=False)

    def eps(self):
        return max(1 / (self.iters + 1), self.min_eps)

    def select(self, player: int, choices: List[Card], vals: List[float]):
        '''Epsilon-greedy action selection'''
        if self.train and np.random.rand() < self.eps():
            return np.random.choice(choices)

        if player == 0:
            return choices[np.argmax(vals)]
        else:
            return choices[np.argmin(vals)]

    def makeDecision(self, s: State, response: DecisionResponse):
        d: DecisionState = s.decision
        p: int = s.player
        if s.phase == Phase.ActionPhase:
            assert False, 'MCTS does not support action cards yet'
        elif s.phase == Phase.TreasurePhase:
            response.single_card = d.card_choices[0]
        else:
            vals = []
            choices = d.card_choices + [None]

            X = s.lookahead_batch_featurize(choices)
            vals = self.model(X).detach().cpu().numpy()

            choice = self.select(p, choices, vals)
            response.single_card = choice


# TODO: Expand MCTS to work outside of sandbox games
class MCTSPlayer(Player):
    def __init__(self, rollout, tree: GameTree, use_tree=True):
        self.tree: GameTree = tree
        self.rollout: RolloutModel = rollout
        self.heuristic = PlayerHeuristic(RandomBuyAgenda())
        self.use_tree = use_tree

    @classmethod
    def load(cls, **kwargs):
        tree: GameTree = kwargs.pop('tree')
        rollout_path: str = kwargs.pop('rollout_path')
        rollout_type: str = kwargs.pop('rollout_type')
        use_tree: bool = kwargs.pop('use_tree')
        # TODO: Fix this to work with other rollout models.
        try:
            rollout_model = load_rollout(rollout_type=rollout_type, model=rollout_path)
        except ImportError:
            logging.error(f'Failed to load rollout from {rollout_path}, defaulting to random rollouts.')
            rollout_model = RandomRollout()

        return cls(rollout=rollout_model, tree=tree, use_tree=use_tree)

    def makeDecision(self, s: State, response: DecisionResponse):
        d: DecisionState = s.decision
        if s.phase == Phase.ActionPhase:
            if not d.active_card:
                self.heuristic.makeGreedyActionDecision(s, response)
            elif s.events:
                event = s.events[-1]
                if isinstance(event, DiscardDownToN):
                    self.heuristic.makeDiscardDownDecision(s, response)
                elif isinstance(event, MoatReveal):
                    self.heuristic.makeBaseDecision(s, response)
                else:
                    raise ValueError(f'Event {type(event)} not supported')
            else:
                self.heuristic.makeBaseDecision(s, response)

        elif s.phase == Phase.TreasurePhase:
            response.single_card = d.card_choices[0]
        else:
            # Remove Curse
            choices = list(filter(lambda x: not isinstance(x, Curse), d.card_choices + [None]))

            # Rollout (out-of-tree) case; tree actually isn't that good
            if not self.tree.in_tree or not self.use_tree:
                logging.log(level=BUY, msg='Rollout')
                response.single_card = self.rollout.select(choices, state=s)
                return

            # the next node in the tree is the one that maximizes the UCB1 score
            try:
                # Remove Copper and Victory cards -- tree never gets that deep anyways
                tree_choices =  list(filter(lambda x: not isinstance(x, Copper) and not issubclass(type(x), VictoryCard), choices))
                card = self.tree.select(tree_choices)
                logging.log(level=BUY, msg=f'Selection: {self.tree.node.n}')
            except ValueError:
                card = self.rollout.select(choices, state=s)

            response.single_card = card


class RolloutPlayer(Player):
    def __init__(self, rollout: RolloutModel):
        self.rollout = rollout

    @classmethod
    def load(cls, **kwargs):
        rollout_path: str = kwargs.pop('rollout_path')
        rollout_type: str = kwargs.pop('rollout_type')

        # TODO: Fix this to work with other rollout models.
        try:
            rollout_model = load_rollout(rollout_type=rollout_type, model=rollout_path)
        except ImportError:
            logging.error(f'Failed to load rollout from {rollout_path}, defaulting to random rollouts.')
            rollout_model = RandomRollout()
        return cls(rollout_model)

    def makeDecision(self, s: State, response: DecisionResponse):
        d: DecisionState = s.decision
        if s.phase == Phase.ActionPhase:
            assert False, 'MCTS does not support action cards yet'
        elif s.phase == Phase.TreasurePhase:
            response.single_card = d.card_choices[0]
        else:
            choices = d.card_choices + [None]

            # the next node in the tree is the one that maximizes the UCB1 score
            card = self.rollout.select(choices, state=s)

            response.single_card = card


class HeuristicPlayer(Player):
    def __init__(self, agenda: BuyAgenda, train=False):
        self.heuristic = PlayerHeuristic(agenda)
        self.train = train

    @classmethod
    def load(cls, train=False, **kwargs):
        return cls(agenda=kwargs.pop('agenda'), train=train)

    def makePhaseDecision(self, s: State, response: DecisionResponse):
        d: DecisionState = s.decision
        player = d.controlling_player
        if s.phase == Phase.ActionPhase:
            self.heuristic.makeGreedyActionDecision(s, response)
        elif s.phase == Phase.TreasurePhase:
            response.single_card = d.card_choices[0]
        else:
            if not self.train:
                remove_first_card(Curse(), d.card_choices)
            response.single_card = self.heuristic.agenda.buy(s, player, d.card_choices)
        return

    def makeDecision(self, s: State, response: DecisionResponse):
        d: DecisionState = s.decision
        player: int = d.controlling_player
        if d.type != DecisionType.DecisionSelectCards and d.type != DecisionType.DecisionDiscreteChoice:
            logging.error('Invalid decision type')
        if not d.active_card:
            self.makePhaseDecision(s, response)
        elif s.events:
            event = s.events[-1]
            if isinstance(event, PutOnDeckDownToN):
                self.heuristic.makePutDownOnDeckDecision(s, response)
            elif isinstance(event, DiscardDownToN):
                self.heuristic.makeDiscardDownDecision(s, response)
            elif isinstance(event, RemodelExpand):
                if not event.trashed_card:
                    def scoringFunction(card: Card):
                        if isinstance(card, Curse):
                            return 19
                        elif isinstance(card, Estate):
                            return 18
                        elif isinstance(card, VictoryCard):
                            return -200 + card.get_coin_cost()
                        return -card.get_coin_cost()
                    response.cards = heuristic_select_cards(d.card_choices, d.min_cards, scoringFunction)
                else:
                    response.cards.append(self.heuristic.agenda.forceBuy(s, player, d.card_choices))
        else:
            self.heuristic.makeBaseDecision(s, response)


class RandomPlayer(Player):
    def __init__(self, train: bool = False):
        self.train = train

    @classmethod
    def load(cls, **kwargs):
        return cls(train=kwargs['train'])

    def makeDecision(self, s: State, response: DecisionResponse):
        d: DecisionState = s.decision

        # Do not allow RandomPlayer to purchase curses
        if s.phase == Phase.BuyPhase and not self.train:
            remove_first_card(Curse(), d.card_choices)

        # Ensure random player plays all treasures
        if s.phase == Phase.TreasurePhase:
            response.single_card = d.card_choices[0]
            return

        if d.type == DecisionType.DecisionSelectCards:
            cards_to_pick = d.min_cards
            if d.max_cards > d.min_cards:
                cards_to_pick = random.randint(d.min_cards, d.max_cards)

            response.cards = random.sample(d.card_choices, k=min(cards_to_pick, len(d.card_choices)))
        elif d.type == DecisionType.DecisionDiscreteChoice:
            response.choice = random.randint(0, d.min_cards)
        else:
            logging.error('Invalid decision type')

    def __str__(self):
        return 'Random Player'


class HumanPlayer(Player):
    @classmethod
    def load(cls, **kwargs):
        return cls()

    def makeDecision(self, s: State, response: DecisionResponse):
        d: DecisionState = s.decision
        if s.phase == Phase.TreasurePhase:
            response.single_card = d.card_choices[0]
            return

        if d.type == DecisionType.DecisionSelectCards:
            cardsToPick = -1
            d.print_card_choices()
            while (cardsToPick < d.min_cards or cardsToPick > d.max_cards):
                text = ''
                while not text:
                    text = input(f'Pick between {d.min_cards} and {d.max_cards} of the above cards:\n')
                cardsToPick = int(text)

            responseIdxs = []
            for i in range(cardsToPick):
                cardIdx = -1
                while (cardIdx == -1 or cardIdx in responseIdxs or cardIdx >= len(d.card_choices)):
                    d.print_card_choices()
                    text = ''
                    while not text:
                        text = input('Choose another card:\n')
                    cardIdx = int(text)
                responseIdxs.append(cardIdx)
                response.cards.append(d.card_choices[cardIdx])
        elif d.type == DecisionType.DecisionDiscreteChoice:
            choice = -1
            while choice == -1 or choice > d.min_cards:
                text = ''
                while not text:
                    text = input('Please make a discrete choice from the above cards:\n')
                choice = int(text)
                d.print_card_choices()
            response.choice = choice
        else:
            logging.error(f'Player {s.player} given invalid decision type.')

    def __str__(self):
        return "Human Player"


class PlayerInfo:
    def __init__(self, id: int, controller: Player):
        self.id = id
        self.controller = controller

    def __str__(self):
        return f'{self.controller} {self.id}'


def load_players(player_types: List[str], models: List[str], train=False, **kwargs) -> List[Player]:
    players = []
    for p_type in player_types:
        p_type_lower = p_type.lower()
        if p_type_lower == 'r':
            players.append(HeuristicPlayer.load(agenda=RandomBuyAgenda(), train=train))
        elif p_type_lower == 'bm':
            players.append(HeuristicPlayer.load(agenda=BigMoneyBuyAgenda(), train=train))
        elif p_type_lower == 'tdbm':
            players.append(HeuristicPlayer.load(agenda=TDBigMoneyBuyAgenda(), train=train))
        elif p_type_lower == 'tdebm':
            players.append(HeuristicPlayer.load(agenda=TDEBigMoneyBuyAgenda(), train=train))
        elif p_type_lower == 'dw':
            players.append(HeuristicPlayer.load(agenda=DoubleWitchBuyAgenda(), train=train))
        elif p_type_lower == 'uct':
            players.append(MCTSPlayer.load(tree=kwargs.pop('tree'), rollout_type=kwargs.pop('rollout_type'), rollout_path=models.pop(0), use_tree=kwargs.pop('use_tree')))
        elif p_type_lower == 'mlp':
            players.append(PredictorMLPPlayer.load(path=models.pop(0), **kwargs))
        elif p_type_lower == 'log':
            players.append(GreedyLogisticPlayer.load(path=models.pop(0)))
        elif p_type_lower == 'gmlp':
            players.append(GreedyMLPPlayer.load(path=models.pop(0)))
        elif p_type_lower == 'h':
            players.append(HumanPlayer())

    if models:
        logging.warning(f'Possible extraneous model paths passed. Remaining paths: {models}')

    return players


def init_players(player_types: List[str], train=True, **kwargs) -> List[Player]:
    players = []

    for p_type in player_types:
        p_type_lower = p_type.lower()
        if p_type_lower == 'r':
            players.append(HeuristicPlayer(agenda=RandomBuyAgenda(), train=train))
        elif p_type_lower == 'bm':
            players.append(HeuristicPlayer(agenda=BigMoneyBuyAgenda(), train=train))
        elif p_type_lower == 'tdbm':
            players.append(HeuristicPlayer(agenda=TDBigMoneyBuyAgenda(), train=train))
        elif p_type_lower == 'tdebm':
            players.append(HeuristicPlayer(agenda=TDEBigMoneyBuyAgenda(), train=train))
        elif p_type_lower == 'dw':
            players.append(HeuristicPlayer(agenda=DoubleWitchBuyAgenda()))
        elif p_type_lower == 'uct':
            # there will ever only be one MCTSPlayer initialized (as opposed to loaded) since the game tree is shared
            rollout = init_rollouts(rollout_types=[kwargs.pop('rollout_type')], **kwargs)[0]
            players.append(MCTSPlayer(rollout=rollout, tree=kwargs.pop('tree')))

    return players
