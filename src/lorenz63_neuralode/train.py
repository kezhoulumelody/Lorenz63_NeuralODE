"""Training utilities for Lorenz63 structured Neural ODE learning."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW

from .data import get_dataloaders
from .eval import compute_tendency_rmse
from .models import build_lorenz63_model


def train_lorenz63_residual(
    npz_path: str | Path = "data/lorenz63_trajectory_10-28-2.7.npz",
    output_path: str | Path = "outputs/lorenz63_residual_model.pt",
    model_name: str = "residual_mlp",
    sigma_init: float = 10.0,
    rho_init: float = 28.0,
    beta_init: float = 2.7,
    hidden_dim: int = 64,
    n_hidden_layers: int = 2,
    batch_size: int = 256,
    n_epochs: int = 200,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    device: str = "cpu",
) -> tuple[torch.nn.Module, dict[str, list[float]]]:
    """Train A_theta and G_theta(x) to match Lorenz63 derivatives."""
    train_loader, test_loader = get_dataloaders(
        npz_path=npz_path,
        batch_size=batch_size,
    )

    model = build_lorenz63_model(
        model_name=model_name,
        sigma_init=sigma_init,
        rho_init=rho_init,
        beta_init=beta_init,
        hidden_dim=hidden_dim,
        n_hidden_layers=n_hidden_layers,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    history = {"train_rmse": [], "test_rmse": []}

    for epoch in range(1, n_epochs + 1):
        model.train()
        total_loss = 0.0
        n_samples = 0

        for batch in train_loader:
            state = batch["state"].to(device)
            target_dxdt = batch["dxdt"].to(device)

            pred_dxdt = model(state)
            loss = loss_fn(pred_dxdt, target_dxdt)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            batch_size_actual = state.shape[0]
            total_loss += loss.item() * batch_size_actual
            n_samples += batch_size_actual

        train_rmse = (total_loss / max(n_samples, 1)) ** 0.5
        test_rmse = compute_tendency_rmse(model, test_loader, device=device)
        history["train_rmse"].append(float(train_rmse))
        history["test_rmse"].append(float(test_rmse))

        print(
            f"Epoch {epoch:04d} | "
            f"train tendency RMSE: {train_rmse:.6f} | "
            f"test tendency RMSE: {test_rmse:.6f}"
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "history": history,
            "params": {
                "model_name": model_name,
                "sigma_init": sigma_init,
                "rho_init": rho_init,
                "beta_init": beta_init,
                "learned_A": model.A.detach().cpu() if hasattr(model, "A") else None,
                "hidden_dim": hidden_dim,
                "n_hidden_layers": n_hidden_layers,
            },
        },
        output_path,
    )
    print(f"Saved model checkpoint to {output_path}")

    return model, history


def parse_args():
    parser = argparse.ArgumentParser(description="Train Lorenz63 structured Neural ODE model.")
    parser.add_argument("--data", default="data/lorenz63_trajectory_10-28-2.7.npz")
    parser.add_argument("--output", default="outputs/lorenz63_residual_model.pt")
    parser.add_argument(
        "--model",
        default="residual_mlp",
        choices=[
            "linear",
            "polynomial",
            "residual_mlp",
            "residual_mix",
            "pure_mlp",
            "bilinear",
            "attention",
            "graph",
            "transformer",
        ],
    )
    parser.add_argument("--sigma-init", type=float, default=10.0)
    parser.add_argument("--rho-init", type=float, default=28.0)
    parser.add_argument("--beta-init", type=float, default=2.7)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--n-hidden-layers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    train_lorenz63_residual(
        npz_path=args.data,
        output_path=args.output,
        model_name=args.model,
        sigma_init=args.sigma_init,
        rho_init=args.rho_init,
        beta_init=args.beta_init,
        hidden_dim=args.hidden_dim,
        n_hidden_layers=args.n_hidden_layers,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=args.device,
    )


if __name__ == "__main__":
    main()
