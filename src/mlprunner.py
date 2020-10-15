from argparse import ArgumentParser

import torch
from torch.autograd import Variable
from tqdm import tqdm

from aiconfig import model_dir
from config import GameConfig
from constants import SANDBOX_CARDS
from enums import DecisionType, Phase, StartingSplit
from game import Game
from mlp import SandboxMLP
from player import MLPPlayer, Player
from state import DecisionResponse, DecisionState, State


class MLP:
    def __init__(self, n: int, l: int, tol: float, dtype, **kwargs):
        n_players = 2
        # None option, number of turns, and score
        n_extra = 2
        D_in, D_out = n_players * (len(SANDBOX_CARDS) + n_extra) + len(SANDBOX_CARDS), 1
        H = (D_in + D_out) // 2
        # Setup network
        self.model = SandboxMLP(D_in, H, D_out)
        self.model.cuda()
        self.config = GameConfig(split=StartingSplit.StartingRandomSplit, prosperity=False, num_players=2, sandbox=True)
        self.cards = [card_class() for card_class in SANDBOX_CARDS]
        player = MLPPlayer(self.model, self.cards, 2)
        self.players = [player, player]
        self.game = Game(self.config, self.players)
        self.n = n
        self.l = l
        self.tol = tol
        self.dtype = dtype 

        self.criterion = torch.nn.L1Loss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), **kwargs)

    def reset(self):
        self.game = Game(self.config, self.players)
        self.game.new_game()
        self.game.state.advance_next_decision()
        for player in self.players:
            player.reset()


    def train(self):
        for i in tqdm(range(self.n)):
            # Setup game
            self.reset()
            s: State = self.game.state
            d: DecisionState = s.decision

            last_x = torch.randn(self.model.D_in).cuda()
            while d.type != DecisionType.DecisionGameOver:
                response = DecisionResponse([])
                p: MLPPlayer = self.game.players[d.controlling_player].controller
                train = s.phase == Phase.BuyPhase

                if train: 
                    x = p.featurize(s)
                    tgt = self.model(x)

                    self.optimizer.zero_grad()
                    y = self.model(last_x)
                    loss = self.criterion(y, tgt)
                    loss.backward()
                    self.optimizer.step()

                    last_x = x

                # Player 0(1) makes decision
                p.makeDecision(s, response)
                s.process_decision(response)

                # Player 1(0)'s turn
                s.advance_next_decision()

            tgt = torch.FloatTensor([1]).cuda() if s.is_winner(s.player) else torch.FloatTensor([0]).cuda()

            self.optimizer.zero_grad()
            y = self.model(last_x)
            loss = self.criterion(y, tgt)
            loss.backward()
            self.optimizer.step()

            if i > 0 and i % self.l == 0:
                print(f'Epoch {i} loss: {loss}')
                # print(f'Epoch {i} loss: {loss} | P0: {s.get_player_score(0)} | P1: {s.get_player_score(1)} | {y.item()} | {tgt.item()}\n {x}')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-n', default=1000, type=int, help='Number of training iterations')
    parser.add_argument('-l', default=10, type=int, help='Number of iterations before logging')
    parser.add_argument('--lr', default=1e-3, type=float, help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.01)
    parser.add_argument('--decay', type=float, default=1e-4)
    parser.add_argument('--cuda', action='store_true', help='Whether or not to use GPU')
    parser.add_argument('--save', action='store_true', help='Whether or not to save the model.')
    parser.add_argument('--path', type=str, help='Where to save the model', default=model_dir)
    parser.add_argument('--tol', type=float, default=1e-10, help='Min difference in loss to stop training')


    args = parser.parse_args()

    dtype = torch.cuda.FloatTensor if args.cuda else torch.FloatTensor
    mlp = MLP(args.n, args.l, args.tol, dtype, momentum=args.momentum, lr=args.lr, weight_decay=args.decay)
    mlp.train()

    if args.save: 
        torch.save(mlp.model.state_dict(), args.path)
