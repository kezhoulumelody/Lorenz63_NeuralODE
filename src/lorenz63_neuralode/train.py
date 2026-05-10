"""Training utilities for Lorenz63 structured Neural ODE learning."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW

from .data import get_dataloaders
from .eval import compute_tendency_component_rmse, compute_tendency_rmse
from .models import build_lorenz63_model


def save_learning_curve(history: dict[str, list[float]], output_path: str | Path) -> None:
    """Save overall and per-component train/test RMSE curves."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(history["train_rmse"]) + 1)
    component_names = ("dx/dt", "dy/dt", "dz/dt")
    colors = ("tab:blue", "tab:orange", "tab:green")

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    axes[0].plot(epochs, history["train_rmse"], label="train", color="black", linewidth=1.8)
    axes[0].plot(
        epochs,
        history["test_rmse"],
        label="test",
        color="tab:red",
        linewidth=1.8,
        linestyle="--",
    )
    axes[0].set_ylabel("Overall RMSE")
    axes[0].set_title("Lorenz63 tendency learning curve")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    train_components = torch.tensor(history["train_component_rmse"])
    test_components = torch.tensor(history["test_component_rmse"])
    for idx, (name, color) in enumerate(zip(component_names, colors, strict=True)):
        axes[1].plot(
            epochs,
            train_components[:, idx],
            label=f"train {name}",
            color=color,
            linewidth=1.6,
        )
        axes[1].plot(
            epochs,
            test_components[:, idx],
            label=f"test {name}",
            color=color,
            linewidth=1.6,
            linestyle="--",
        )

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Component RMSE")
    axes[1].legend(ncol=2)
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


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
    curve_output_path: str | Path | None = None,
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
    history = {
        "train_rmse": [],
        "test_rmse": [],
        "train_component_rmse": [],
        "test_component_rmse": [],
    }

    for epoch in range(1, n_epochs + 1):
        model.train()
        total_loss = 0.0
        total_component_sse = torch.zeros(3, device=device)
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
            total_component_sse += torch.sum((pred_dxdt.detach() - target_dxdt) ** 2, dim=0)
            total_loss += loss.item() * batch_size_actual
            n_samples += batch_size_actual

        train_rmse = (total_loss / max(n_samples, 1)) ** 0.5
        train_component_rmse = torch.sqrt(total_component_sse / max(n_samples, 1)).detach().cpu()
        test_rmse = compute_tendency_rmse(model, test_loader, device=device)
        test_component_rmse = compute_tendency_component_rmse(model, test_loader, device=device)
        history["train_rmse"].append(float(train_rmse))
        history["test_rmse"].append(float(test_rmse))
        history["train_component_rmse"].append([float(value) for value in train_component_rmse])
        history["test_component_rmse"].append([float(value) for value in test_component_rmse])

        print(
            f"Epoch {epoch:04d} | "
            f"train tendency RMSE: {train_rmse:.6f} | "
            f"test tendency RMSE: {test_rmse:.6f} | "
            f"train xyz: {train_component_rmse[0]:.6f}, "
            f"{train_component_rmse[1]:.6f}, {train_component_rmse[2]:.6f} | "
            f"test xyz: {test_component_rmse[0]:.6f}, "
            f"{test_component_rmse[1]:.6f}, {test_component_rmse[2]:.6f}"
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

    if curve_output_path is None:
        curve_output_path = output_path.with_name(f"{output_path.stem}_learning_curve.png")
    save_learning_curve(history, curve_output_path)
    print(f"Saved learning curve to {curve_output_path}")

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
            "pure_transformer",
            "bilinear",
            "bilinear_graph",
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
    parser.add_argument(
        "--curve-output",
        default=None,
        help="Path for the learning curve PNG. Defaults to output stem plus '_learning_curve.png'.",
    )
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
        curve_output_path=args.curve_output,
    )


if __name__ == "__main__":
    main()
