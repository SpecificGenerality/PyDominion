from argparse import ArgumentParser

import torch
from torch.autograd import Variable
from tqdm import tqdm

from config import GameConfig
from constants import SANDBOX_CARDS
from enums import DecisionType, StartingSplit
from game import Game
from mlp import SandboxMLP
from player import MLPPlayer, Player
from state import DecisionResponse, DecisionState, State


class MLP:
    def __init__(self, n: int, lr: float, l: int):
        device = torch.device("cuda:0")

        n_players = 2
        # number of turns and score
        n_extra = 2
        D_in, D_out = n_players * (len(SANDBOX_CARDS) + n_extra), 1
        H = 2 * D_in
        # Setup network
        self.model = SandboxMLP(D_in, H, D_out)
        self.model.cuda()
        self.config = GameConfig(split=StartingSplit.StartingRandomSplit, prosperity=False, num_players=2, sandbox=True)
        self.cards = [card_class() for card_class in SANDBOX_CARDS]
        self.players = [MLPPlayer(self.model, self.cards, 2), MLPPlayer(self.model, self.cards, 2)]
        self.game = Game(self.config, self.players)
        self.n = n
        self.lr = lr
        self.l = l

        self.criterion = torch.nn.BCELoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)

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

            last_x = None
            while d.type != DecisionType.DecisionGameOver:
                response = DecisionResponse([])
                p: MLPPlayer = self.game.players[d.controlling_player].controller
                p.makeDecision(s, response)
                s.process_decision(response)
                s.advance_next_decision()
                x = p.featurize(s, lookahead_card=None)
                tgt = self.model(x)
                X, Y = Variable(x, requires_grad=True), Variable(tgt, requires_grad=False)

                if last_x is not None:
                    self.optimizer.zero_grad()
                    y = self.model(last_x)
                    loss = self.criterion(y, Y)
                    loss.backward()
                    self.optimizer.step()

                last_x = x

            p: MLPPlayer = self.game.players[d.controlling_player].controller
            x = p.featurize(s, lookahead_card=None)
            p_id: int = self.game.players[d.controlling_player].id
            tgt = torch.FloatTensor([1]) if s.is_winner(p_id) else torch.FloatTensor([0])
            self.optimizer.zero_grad()
            y = self.model.forward(x)
            loss = self.criterion(y, tgt)
            loss.backward()
            self.optimizer.step()

            if i > 0 and i % self.l == 0:
                print(f'Epoch {i} loss: {loss}')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-n', default=10000, type=int, help='Number of training iterations')
    parser.add_argument('-l', default=100, type=int, help='Number of iterations before logging')
    parser.add_argument('--lr', default=1e-4, type=float, help='Learning rate')

    args = parser.parse_args()

    mlp = MLP(args.n, args.lr, args.l)
    mlp.train()
