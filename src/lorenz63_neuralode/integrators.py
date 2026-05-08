"""Simple integrators for learned Lorenz63 dynamics."""

from __future__ import annotations

import torch


def rk4_step(model, x: torch.Tensor, dt: float) -> torch.Tensor:
    """One fixed-step fourth-order Runge-Kutta update."""
    k1 = model(x)
    k2 = model(x + 0.5 * dt * k1)
    k3 = model(x + 0.5 * dt * k2)
    k4 = model(x + dt * k3)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def euler_step(model, x: torch.Tensor, dt: float) -> torch.Tensor:
    """One forward Euler update."""
    return x + dt * model(x)
