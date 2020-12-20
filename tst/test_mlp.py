import unittest

import numpy as np
import torch
from src.mlp import SandboxPerceptron


class TestPerceptron(unittest.TestCase):
    def setUp(self):
        self.perceptron = SandboxPerceptron(D_in=2, lr=1, gamma=1, lambd=1, device='cpu')
        self.perceptron.fc1.weight.data.fill_(0.5)
        self.perceptron.fc1.bias.data.fill_(0.5)
        self.perceptron.init_eligibility_traces()

    def test_update_weights(self):
        x = torch.tensor([1., 0.])
        o = self.perceptron(x)

        p_next = torch.tensor([1.])
        delta = p_next - o

        self.perceptron.update_weights(o, p_next)

        grad_constant = o * (1 - o)
        weights = torch.FloatTensor([[0.5, 0.5]]) + grad_constant * delta * torch.FloatTensor([[1, 0]])
        bias = torch.FloatTensor([0.5]) + grad_constant * delta * torch.FloatTensor([1])
        np.testing.assert_allclose(self.perceptron.fc1.weight.data.detach().numpy(), weights.detach().numpy())
        np.testing.assert_allclose(self.perceptron.fc1.bias.data.detach().numpy(), bias.detach().numpy())
