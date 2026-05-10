"""Evaluation helpers for Lorenz63 Neural ODE models."""

from __future__ import annotations

import torch


@torch.no_grad()
def compute_tendency_rmse(model, dataloader, device: str = "cpu") -> float:
    """Compute RMSE for the full learned tendency A_theta x + G_theta(x)."""
    model.eval()
    total_mse = 0.0
    n_samples = 0

    for batch in dataloader:
        state = batch["state"].to(device)
        target_dxdt = batch["dxdt"].to(device)
        pred_dxdt = model(state)
        mse = torch.mean((pred_dxdt - target_dxdt) ** 2)

        batch_size = state.shape[0]
        total_mse += mse.item() * batch_size
        n_samples += batch_size

    return float((total_mse / max(n_samples, 1)) ** 0.5)


@torch.no_grad()
def compute_tendency_component_rmse(model, dataloader, device: str = "cpu") -> torch.Tensor:
    """Compute per-component RMSE for dx/dt, dy/dt, and dz/dt."""
    model.eval()
    total_sse = torch.zeros(3, device=device)
    n_samples = 0

    for batch in dataloader:
        state = batch["state"].to(device)
        target_dxdt = batch["dxdt"].to(device)
        pred_dxdt = model(state)
        total_sse += torch.sum((pred_dxdt - target_dxdt) ** 2, dim=0)
        n_samples += state.shape[0]

    return torch.sqrt(total_sse / max(n_samples, 1)).detach().cpu()


compute_residual_rmse = compute_tendency_rmse


@torch.no_grad()
def euler_rollout(model, initial_state: torch.Tensor, dt: float, n_steps: int) -> torch.Tensor:
    """Roll out the learned Lorenz63 dynamics with forward Euler."""
    model.eval()
    states = [initial_state]
    current = initial_state

    for _ in range(n_steps):
        current = current + dt * model(current)
        states.append(current)

    return torch.stack(states, dim=0)
