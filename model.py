import torch.nn as nn


class PlainStack(nn.Module):
    """Deep stack with NO weight sharing, NO skip connection, NO dt scaling.

    h_{k+1} = f_theta_k(h_k), each layer independently parameterized.
    """

    def __init__(self, dim, hidden, depth):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(dim, hidden), nn.Tanh(), nn.Linear(hidden, dim))
            for _ in range(depth)
        ])

    def forward(self, h):
        for layer in self.layers:
            h = layer(h)
        return h


class PlainClassifier(nn.Module):
    def __init__(self, in_dim=2, hidden_dim=8, mlp_hidden=32, depth=8):
        super().__init__()
        self.embed = nn.Linear(in_dim, hidden_dim)
        self.stack = PlainStack(hidden_dim, mlp_hidden, depth)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        return self.head(self.stack(self.embed(x))).squeeze(-1)
