from argparse import ArgumentParser

import torch
from tqdm import tqdm

from aiconfig import model_dir
from config import GameConfig
from enums import DecisionType, Phase, StartingSplit
from game import Game
from mlp import SandboxMLP, SandboxPerceptron
from player import MLPPlayer
from state import DecisionResponse, DecisionState, State
from torch.utils.tensorboard import SummaryWriter


class MLP:
    def __init__(self, n: int, **kwargs):
        # Configure game
        self.config = GameConfig(split=StartingSplit.StartingRandomSplit, prosperity=False, num_players=2, sandbox=True)
        self.tensorboard_logging = kwargs.pop('tensorboard_logging')

        # Define network parameters
        self.D_in, self.D_out = self.config.feature_size, 1
        self.H = (self.D_in + self.D_out) // 2

        # Initialize network
        self.model = SandboxMLP(self.D_in, self.H, self.D_out, **kwargs)
        self.model.cuda()
        torch.nn.init.xavier_uniform_(self.model.fc1.weight)
        self.model.init_eligibility_traces()
        self.criterion = torch.nn.MSELoss()

        # Initialize players
        player = MLPPlayer(self.model)
        self.players = [player, player]

        # Initalize game
        self.game = Game(self.config, self.players)
        self.n = n

        self.writer = SummaryWriter()

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

            last_x = torch.randn(self.D_in).cuda()
            while d.type != DecisionType.DecisionGameOver:
                response = DecisionResponse([])
                p: MLPPlayer = self.game.players[d.controlling_player].controller
                train = s.phase == Phase.BuyPhase

                if train:
                    x = p.featurize(s)
                    tgt = self.model(x)

                    y = self.model(last_x)
                    self.model.update_weights(y, tgt)
                    last_x = x

                # Player 0(1) makes decision
                p.makeDecision(s, response)
                s.process_decision(response)

                # Player 1(0)'s turn
                s.advance_next_decision()

            if s.is_winner(0) and s.is_winner(1):
                tgt = torch.FloatTensor([1]).cuda()
            elif s.is_winner(0):
                tgt = torch.FloatTensor([1]).cuda()
            else:
                tgt = torch.FloatTensor([0]).cuda()

            # self.optimizer.zero_grad()
            y = self.model(last_x)
            self.model.update_weights(y, tgt)
            loss = self.criterion(y, tgt)
            if self.tensorboard_logging:
                self.writer.add_scalar("Loss/train", loss, i)
            p.iters += 1


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-n', default=1000, type=int, help='Number of training iterations')
    parser.add_argument('--lr', default=1e-5, type=float, help='Learning rate')
    parser.add_argument('--lambd', default=0.5, type=float, help='Lambda parameter for TD-learning')
    parser.add_argument('--cuda', action='store_true', help='Whether or not to use the GPU')
    parser.add_argument('--tensorboard', action='store_true', help='Whether or not to log using TensorBoard')
    parser.add_argument('--save', action='store_true', help='Whether or not to save the model.')
    parser.add_argument('--path', type=str, help='Where to save the model', default=model_dir)

    args = parser.parse_args()

    device = 'cuda' if args.cuda else 'cpu'
    mlp = MLP(args.n, device=device, lr=args.lr, lambd=args.lambd, tensorboard_logging=args.tensorboard)
    mlp.train()
    mlp.writer.flush()

    if args.save:
        torch.save(mlp.model.state_dict(), args.path)
