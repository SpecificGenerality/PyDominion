import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class SandboxMLP(nn.Module):
    def __init__(self, D_in: int, H, D_out):
        super(SandboxMLP, self).__init__()
        self.D_in = D_in
        self.fc1 = nn.Linear(D_in, H)
        self.fc2 = nn.Linear(H, D_out)

    def forward(self, x):
         x = torch.sigmoid(self.fc2(F.relu(self.fc1(x))))
         return x
