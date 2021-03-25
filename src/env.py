import logging
from abc import ABC, abstractmethod
from typing import Any, Iterable, Tuple

from config import GameConfig
from constants import ACTION, BUY
from enums import Phase
from game import Game
from player import HumanPlayer, Player
from state import DecisionResponse, DecisionState, State


class Environment(ABC):
    def __init__(self, config: GameConfig, players: Iterable[Player], logger):
        self.config = config
        self.players = players
        self.game = Game(config, players)
        self.logger = logger

        logging.addLevelName(ACTION, 'ACTION')
        logging.addLevelName(BUY, 'BUY')

        if any(isinstance(player, HumanPlayer) for player in players):
            logging.basicConfig(format='%(message)s', level=logging.INFO)
        else:
            logging.basicConfig(format='%(levelname)s:%(message)s')

    @abstractmethod
    def reset(self) -> State:
        '''Reset the environment for another training epoch'''
        pass

    @abstractmethod
    def step(self, action: DecisionResponse) -> Tuple[State, int, bool, Any]:
        '''
            Given an integral choice return the following tuple:
            (observation, reward, done, info)
        '''
        pass


class FullEnvironment(Environment):
    def __init__(self, config: GameConfig, players: Iterable[Player], logger=logging.getLogger()):
        super().__init__(config, players, logger)

    def reset(self, **kwargs):
        self.game = Game(self.config, self.players)
        self.game.new_game()
        self.game.state.advance_next_decision()

        return self.game.state

    def step(self, action: DecisionResponse) -> Tuple[State, int, bool, Any]:
        s: State = self.game.state

        s.process_decision(action)
        s.advance_next_decision()

        reward = 0
        if self._done:
            p0win = self.game.is_winner(0)
            p1win = self.game.is_winner(1)
            if p0win and p1win:
                reward = 0
            elif p0win:
                reward = 1
            else:
                reward = -1

        return s, reward, self._done, None

    @property
    def _done(self) -> bool:
        return self.game.done


class DefaultEnvironment(Environment):
    def __init__(self, config: GameConfig, players: Iterable[Player], logger=logging.getLogger()):
        super().__init__(config, players, logger)

    def reset(self, **kwargs) -> State:
        self.game = Game(self.config, self.players)
        self.game.new_game()
        self.game.state.advance_next_decision()

        s: State = self.game.state
        d: DecisionState = s.decision

        while s.phase != Phase.BuyPhase and not self._done:
            response = DecisionResponse([])
            p = self.game.players[d.controlling_player].controller
            p.makeDecision(s, response)
            s.process_decision(response)
            s.advance_next_decision()

        return self.game.state

    def step(self, action: DecisionResponse) -> Tuple[State, int, bool, Any]:
        s: State = self.game.state
        d: DecisionState = s.decision

        if s.phase != Phase.BuyPhase:
            raise ValueError('Cannot step from any phase other than Buy Phase.')

        p: Player = self.game.players[d.controlling_player].controller

        s.process_decision(action)

        s.advance_next_decision()

        # Skip all non-Buy phases until end of game
        while s.phase != Phase.BuyPhase and not self._done:
            response = DecisionResponse([])
            p = self.game.players[d.controlling_player].controller
            p.makeDecision(s, response)
            s.process_decision(response)
            s.advance_next_decision()

        reward = 0
        if self._done:
            p0win = self.game.is_winner(0)
            p1win = self.game.is_winner(1)
            if p0win and p1win:
                reward = 0
            elif p0win:
                reward = 1
            else:
                reward = -1

        return s, reward, self._done, None

    @property
    def _done(self) -> bool:
        return self.game.done
