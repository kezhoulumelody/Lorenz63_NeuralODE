"""Generate and load Lorenz63 data for residual learning."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from scipy.integrate import solve_ivp
from scipy.integrate._ivp.rk import DOP853 # new import
from torch.utils.data import DataLoader, Dataset, random_split

def lorenz63(t, state, sigma=10.0, rho=28.0, beta=8.0 / 3.0, forcing=(0.0, 0.0, 0.0)): # we add Wiener process using the Euler-Maruyama scheme in the generate_lorenz63 function
    """Lorenz63 system. New kwargs for forcing with default at 0."""
    x, y, z = state
    Fx, Fy, Fz = forcing

    dxdt = sigma * (y - x) + Fx
    dydt = x * (rho - z) - y + Fy
    dzdt = x * y - beta * z + Fz

    return [dxdt, dydt, dzdt]


def generate_lorenz63(
    t_start=0.0,
    t_end=50.0,
    dt=0.01,
    y0=None,
    sigma=10.0,
    rho=28.0,
    beta=8.0 / 3.0,
    forcing=(0.0, 0.0, 0.0),
    noise=(0.0, 0.0, 0.0),
    noise_seed=None
):
    """Solve Lorenz63 and return t, x, y, z, states, and parameter metadata."""
    if y0 is None:
        y0 = [1.0, 1.0, 1.0]

    t_eval = np.arange(t_start, t_end, dt)
    fun = lambda t, state: lorenz63(t, state, sigma=sigma, rho=rho, beta=beta, forcing=forcing)

    if noise == (0.0, 0.0, 0.0): # with no noise, it solves using standard RK8
        sol = solve_ivp(
            fun=fun,
            t_span=(t_start, t_end),
            y0=y0,
            t_eval=t_eval,
            method="DOP853",
        )

        x, y, z = sol.y
        states = sol.y.T
        t = sol.t

        if not sol.success:
            raise RuntimeError(f"Lorenz63 solve_ivp failed: {sol.message}")


    else: # with noise, it solves with RK8 + Euler-Maruyama scheme 
        rng = np.random.default_rng(noise_seed)
        sqrt_dt = np.sqrt(dt)
        noise_arr = np.array(noise)
        
        # Pre-allocate array for states
        sim_states = np.zeros((len(t_eval), 3))
        sim_states[0] = y0
        y_curr = np.array(y0, dtype=float)
        
        for i in range(1, len(t_eval)):
            t_curr = t_eval[i-1]
            t_next = t_eval[i]
            
            # DOP853 deterministic step
            solver = DOP853(fun, t_curr, y_curr, t_next)
            while solver.status == "running":
                solver.step()
            y_det = solver.y
            
            # Euler-Maruyama stochastic step
            dW = rng.standard_normal(3) * sqrt_dt
            y_curr = y_det + noise_arr * dW

            #save result
            sim_states[i] = y_curr

        x, y, z = sim_states.T
        states = sim_states    
        t = t_eval

        if i < len(t_eval)-1:
            raise RuntimeError(f"Stochastic Lorenz63 solve_ivp failed.")

    return {
        "t": t,
        "x": x,
        "y": y,
        "z": z,
        "states": states,
        "sigma": float(sigma),
        "rho": float(rho),
        "beta": float(beta),
        "forcing": forcing,
        "noise": noise,
        "noise_seed": noise_seed
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
        forcing=data["forcing"],
        noise=data["noise"],
        noise_seed=data["noise_seed"]
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
    parser.add_argument("--forcing", type=float, nargs=3, default=[0.0, 0.0, 0.0])
    parser.add_argument("--noise", type=float, nargs=3, default=[0.0, 0.0, 0.0])
    parser.add_argument("--noise_seed", type=float, default=None)
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
        forcing=tuple(args.forcing),
        noise=tuple(args.noise),
        noise_seed=args.noise_seed
    )
    save_lorenz63_npz(data, args.output)
    print(f"Saved Lorenz63 data to {args.output}")


if __name__ == "__main__":
    main()