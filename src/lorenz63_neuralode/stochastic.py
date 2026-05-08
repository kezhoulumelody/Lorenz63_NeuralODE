"""Optional stochastic rollout helpers for Lorenz63 models."""

from __future__ import annotations

import torch


def stochastic_euler_step(
    model,
    x: torch.Tensor,
    dt: float,
    noise_std: float = 0.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """One Euler-Maruyama-style step with optional additive Gaussian noise."""
    drift = model(x)
    x_next = x + dt * drift

    if noise_std > 0.0:
        noise = torch.randn(
            x.shape,
            dtype=x.dtype,
            device=x.device,
            generator=generator,
        )
        x_next = x_next + (dt**0.5) * noise_std * noise

    return x_next


@torch.no_grad()
def stochastic_rollout(
    model,
    initial_state: torch.Tensor,
    dt: float,
    n_steps: int,
    noise_std: float = 0.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Roll out a Lorenz63 model with optional additive stochastic forcing."""
    model.eval()
    states = [initial_state]
    current = initial_state

    for _ in range(n_steps):
        current = stochastic_euler_step(
            model,
            current,
            dt=dt,
            noise_std=noise_std,
            generator=generator,
        )
        states.append(current)

    return torch.stack(states, dim=0)
