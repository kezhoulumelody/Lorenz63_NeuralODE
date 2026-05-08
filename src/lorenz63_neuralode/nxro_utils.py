"""Small utility helpers for Lorenz63 Neural ODE experiments.

The file name is kept for now because it came from the copied NXRO project, but
the contents are Lorenz63-specific and do not depend on the old ocean workflow.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set common random seeds for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_checkpoint(path: str | Path, device: str = "cpu") -> dict:
    """Load a saved Lorenz63 model checkpoint."""
    return torch.load(path, map_location=device)


def npz_to_states(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load time and [x, y, z] states from a Lorenz63 .npz file."""
    archive = np.load(path)
    t = archive["t"]
    if "states" in archive.files:
        states = archive["states"]
    else:
        states = np.stack([archive["x"], archive["y"], archive["z"]], axis=-1)
    return t, states
