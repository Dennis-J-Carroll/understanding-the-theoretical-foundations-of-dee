import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint


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


class _TimeWrapped(nn.Module):
    """torchdiffeq expects func(t, h); f_theta here is autonomous (no explicit t)."""

    def __init__(self, f):
        super().__init__()
        self.f = f

    def forward(self, t, h):
        return self.f(h)


class NeuralODEBlock(nn.Module):
    """Solves dh/dt = f_theta(h) on [0, T] with an adaptive solver (dopri5, adjoint)."""

    def __init__(self, dim, hidden=32, T=1.0, rtol=1e-3, atol=1e-4):
        super().__init__()
        self.f = VectorField(dim, hidden)
        self.func = _TimeWrapped(self.f)
        self.T = T
        self.rtol = rtol
        self.atol = atol

    def forward(self, h0, t_span):
        out = odeint(self.func, h0, t_span, method="dopri5", rtol=self.rtol, atol=self.atol)
        return out[-1]


class NeuralODEClassifier(nn.Module):
    def __init__(self, in_dim=2, hidden_dim=8, mlp_hidden=32, T=1.0):
        super().__init__()
        self.embed = nn.Linear(in_dim, hidden_dim)
        self.block = NeuralODEBlock(hidden_dim, mlp_hidden, T=T)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x, t_span):
        h0 = self.embed(x)
        hT = self.block(h0, t_span)
        return self.head(hT).squeeze(-1)
