import torch.nn as nn


class VectorField(nn.Module):
    """Autonomous f_theta(h) shared across all discretization steps."""

    def __init__(self, dim, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.Tanh(),
            nn.Linear(hidden, dim),
        )

    def forward(self, h):
        return self.net(h)


def euler_integrate(f, h0, num_steps, T=1.0):
    """h_{k+1} = h_k + dt * f(h_k), dt = T / num_steps (Eq. 2 of the survey)."""
    dt = T / num_steps
    h = h0
    for _ in range(num_steps):
        h = h + dt * f(h)
    return h


class EulerODEBlock(nn.Module):
    def __init__(self, dim, hidden=32, T=1.0):
        super().__init__()
        self.f = VectorField(dim, hidden)
        self.T = T

    def forward(self, h0, num_steps):
        return euler_integrate(self.f, h0, num_steps, self.T)


class MidpointODEBlock(nn.Module):
    """RK2 midpoint step: h_{k+1} = h_k + dt * f(h_k + dt/2 * f(h_k))."""

    def __init__(self, dim, hidden=32, T=1.0):
        super().__init__()
        self.f = VectorField(dim, hidden)
        self.T = T

    def forward(self, h0, num_steps):
        dt = self.T / num_steps
        h = h0
        for _ in range(num_steps):
            k1 = self.f(h)
            k2 = self.f(h + 0.5 * dt * k1)
            h = h + dt * k2
        return h


class ODEClassifier(nn.Module):
    def __init__(self, in_dim=2, hidden_dim=8, mlp_hidden=32, T=1.0, block_cls=EulerODEBlock):
        super().__init__()
        self.embed = nn.Linear(in_dim, hidden_dim)
        self.block = block_cls(hidden_dim, mlp_hidden, T=T)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x, num_steps):
        h0 = self.embed(x)
        hT = self.block(h0, num_steps)
        return self.head(hT).squeeze(-1)
