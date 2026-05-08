"""Generate sample Lorenz63 trajectory data with scipy.solve_ivp."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp


def lorenz63(t, state, sigma=10, rho=28, beta=8 / 3):
    """Lorenz63 system.

    The ``t`` argument is required by solve_ivp, even though the Lorenz63
    equations do not explicitly depend on time.
    """
    x, y, z = state

    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z

    return [dxdt, dydt, dzdt]


def generate_lorenz63(t_start=0, t_end=50, dt=0.01, y0=None, sigma=10, rho=28, beta=8 / 3):
    """Solve Lorenz63 and return a dictionary with t, x, y, z, and states."""
    if y0 is None:
        y0 = [1, 1, 1]

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
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Lorenz63 trajectory data.")
    parser.add_argument("--output", default="/data/kezhoulumelody/Lorenz63_NeuralODE/data/lorenz63_trajectory_default.npz")
    parser.add_argument("--t-start", type=float, default=0)
    parser.add_argument("--t-end", type=float, default=50)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--sigma", type=float, default=10)
    parser.add_argument("--rho", type=float, default=28)
    parser.add_argument("--beta", type=float, default=8 / 3)
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
