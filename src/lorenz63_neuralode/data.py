"""Generate and load Lorenz63 data for residual learning."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from scipy.integrate import solve_ivp
from torch.utils.data import DataLoader, Dataset, random_split

def lorenz63(t, state, sigma=10.0, rho=28.0, beta=8.0 / 3.0):
    """Lorenz63 system."""
    x, y, z = state

    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z

    return [dxdt, dydt, dzdt]


def generate_lorenz63(
    t_start=0.0,
    t_end=50.0,
    dt=0.01,
    y0=None,
    sigma=10.0,
    rho=28.0,
    beta=8.0 / 3.0,
):
    """Solve Lorenz63 and return t, x, y, z, states, and parameter metadata."""
    if y0 is None:
        y0 = [1.0, 1.0, 1.0]

    t_eval = np.arange(t_start, t_end, dt)

    sol = solve_ivp(
        fun=lambda t, state: lorenz63(t, state, sigma=sigma, rho=rho, beta=beta),
        t_span=(t_start, t_end),
        y0=y0,
        t_eval=t_eval,
        method="DOP853",
    )

    if not sol.success:
        raise RuntimeError(f"Lorenz63 solve_ivp failed: {sol.message}")

    x, y, z = sol.y
    states = sol.y.T

    return {
        "t": sol.t,
        "x": x,
        "y": y,
        "z": z,
        "states": states,
        "sigma": float(sigma),
        "rho": float(rho),
        "beta": float(beta),
    }


def save_lorenz63_npz(data, output_path):
    """Save Lorenz63 data in a compact NumPy format for later training."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_path,
        t=data["t"],
        x=data["x"],
        y=data["y"],
        z=data["z"],
        states=data["states"],
        sigma=data["sigma"],
        rho=data["rho"],
        beta=data["beta"],
    )


class Lorenz63ResidualDataset(Dataset):
    """Dataset of Lorenz63 states and derivative targets.

    The primary target is ``dxdt``, estimated from the saved trajectory with
    finite differences. This lets the model learn both the structured linear
    coefficients in ``A_theta`` and the nonlinear residual ``G_theta``.
    """

    def __init__(
        self,
        npz_path: str | Path,
    ) -> None:
        archive = np.load(npz_path)
        t = archive["t"].astype(np.float32)

        if "states" in archive.files:
            states = archive["states"].astype(np.float32)
        else:
            states = np.stack([archive["x"], archive["y"], archive["z"]], axis=-1).astype(np.float32)

        dxdt = np.gradient(states, t, axis=0).astype(np.float32)

        self.t = torch.from_numpy(t)
        self.states = torch.from_numpy(states)
        self.dxdt = torch.from_numpy(dxdt)

    def __len__(self) -> int:
        return self.states.shape[0]

    def __getitem__(self, idx: int):
        return {
            "t": self.t[idx],
            "state": self.states[idx],
            "dxdt": self.dxdt[idx],
        }


def get_dataloaders(
    npz_path: str | Path,
    batch_size: int = 256,
    train_fraction: float = 0.8,
    seed: int = 42,
):
    """Build train/test dataloaders for Lorenz63 tendency learning."""
    dataset = Lorenz63ResidualDataset(npz_path=npz_path)
    n_train = int(train_fraction * len(dataset))
    n_test = len(dataset) - n_train
    generator = torch.Generator().manual_seed(seed)
    train_ds, test_ds = random_split(dataset, [n_train, n_test], generator=generator)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Lorenz63 trajectory data.")
    parser.add_argument("--output", default="data/lorenz63_trajectory_10-28-2.7.npz")
    parser.add_argument("--t-start", type=float, default=0.0)
    parser.add_argument("--t-end", type=float, default=50.0)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--sigma", type=float, default=10.0)
    parser.add_argument("--rho", type=float, default=28.0)
    parser.add_argument("--beta", type=float, default=8.0 / 3.0)
    return parser.parse_args()


def main():
    args = parse_args()
    data = generate_lorenz63(
        t_start=args.t_start,
        t_end=args.t_end,
        dt=args.dt,
        sigma=args.sigma,
        rho=args.rho,
        beta=args.beta,
    )
    save_lorenz63_npz(data, args.output)
    print(f"Saved Lorenz63 data to {args.output}")


if __name__ == "__main__":
    main()
