import torch
import torch.nn as nn


class SandboxMLP(nn.Module):
    def __init__(self, D_in: int, H: int, D_out: int, lr=1e-3, gamma=1, lambd=0.7):
        super(SandboxMLP, self).__init__()
        self.D_in = D_in
        self.D_out = D_out
        self.fc1 = nn.Linear(D_in, H)
        self.fc2 = nn.Linear(H, D_out)
        self.sigmoid = nn.Sigmoid()

        self.eligibility_traces = None
        self.lr = lr
        self.gamma = gamma
        self.lambd = lambd

    def forward(self, x):
        x = self.fc1(x)
        x = self.sigmoid(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

    def init_eligibility_traces(self, device='cuda'):
        self.eligibility_traces = [torch.zeros(weights.shape, requires_grad=False, device=device) for weights in list(self.parameters())]

    def update_weights(self, p: torch.tensor, p_next: torch.tensor):
        delta = p_next - p

        self.zero_grad()

        p.backward(torch.ones(self.D_out).cuda(), retain_graph=True)

        for i, param in enumerate(self.parameters()):
            self.eligibility_traces[i] = self.gamma * self.lambd * self.eligibility_traces[i] + param.grad.data
            new_weights = param.data + self.lr * delta * self.eligibility_traces[i]
            param.data = new_weights.data

        return delta


class SandboxPerceptron(nn.Module):
    def __init__(self, D_in: int, D_out: int, lr=1e-3, gamma=1, lambd=0.7, device='cuda'):
        super(SandboxPerceptron, self).__init__()
        self.D_in = D_in
        self.D_out = D_out
        self.fc1 = nn.Linear(D_in, D_out)
        self.sigmoid = nn.Sigmoid()

        self.eligibility_traces = None
        self.lr = lr
        self.gamma = gamma
        self.lambd = lambd

        self.device = device

    def forward(self, x):
        x = self.fc1(x)
        x = self.sigmoid(x)
        return x

    def init_eligibility_traces(self):
        self.eligibility_traces = [torch.zeros(weights.shape, requires_grad=False, device=self.device) for weights in list(self.parameters())]

    def update_weights(self, p: torch.tensor, p_next: torch.tensor):
        delta = p_next - p

        self.zero_grad()

        p.backward(torch.ones(self.D_out, device=self.device), retain_graph=True)

        for i, param in enumerate(self.parameters()):
            self.eligibility_traces[i] = self.gamma * self.lambd * self.eligibility_traces[i] + param.grad.data
            new_weights = param.data + self.lr * delta * self.eligibility_traces[i]
            param.data = new_weights.data

        return delta
