import torch.nn as nn


class SkipOnlyStack(nn.Module):
    """Skip connection present, but NOT weight-tied and NOT dt-scaled:

    h_{k+1} = h_k + f_theta_k(h_k), independent f_theta_k per layer.
    This is what a normal, practical ResNet actually looks like -- the
    isolation vs. C1 is exactly the skip connection; vs. C4 it's exactly
    the missing skip connection that's restored.
    """

    def __init__(self, dim, hidden, depth):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(dim, hidden), nn.Tanh(), nn.Linear(hidden, dim))
            for _ in range(depth)
        ])

    def forward(self, h):
        for layer in self.layers:
            h = h + layer(h)
        return h


class SkipOnlyClassifier(nn.Module):
    def __init__(self, in_dim=2, hidden_dim=8, mlp_hidden=32, depth=8):
        super().__init__()
        self.embed = nn.Linear(in_dim, hidden_dim)
        self.stack = SkipOnlyStack(hidden_dim, mlp_hidden, depth)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        return self.head(self.stack(self.embed(x))).squeeze(-1)
