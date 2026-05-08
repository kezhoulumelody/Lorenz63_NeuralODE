# Lorenz63 Neural ODE

Neural ordinary differential equation experiments for learning, simulating, and analyzing the Lorenz63 dynamical system.


Potential project goals include:

- Generate Lorenz63 trajectories from known parameters.
- Train Neural ODE models to approximate Lorenz63 dynamics.
- Compare learned trajectories against numerical solvers.
- Visualize attractors, prediction error, and long-horizon behavior.

## Overview

The Lorenz63 system is a classic chaotic dynamical system defined by three coupled ordinary differential equations. This repository is intended to provide a clean workspace for experiments that combine the Lorenz63 model with Neural ODE methods. 
The Lorenz63 system is defined as:

$$
\begin{aligned}
\frac{dx}{dt} &= \sigma (y - x) \\
\frac{dy}{dt} &= \rho x - y - xz \\
\frac{dz}{dt} &= xy - \beta z
\end{aligned}
$$

We can split the dynamics as:

$$
\frac{d\mathbf{x}}{dt} = A_\theta \mathbf{x} + G_\theta(\mathbf{x}, \mathcal{G})
$$

where

$$
\mathbf{x} = [x, y, z]^T
$$

and the linear part is:

$$
A =
\begin{bmatrix}
-\sigma & \sigma & 0 \\
\rho & -1 & 0 \\
0 & 0 & -\beta
\end{bmatrix}
$$

The nonlinear residual contains the following, which will be learned by Neural ODE models:

$$
G(\mathbf{x}) =
\begin{bmatrix}
0 \\
-xz \\
xy
\end{bmatrix}
$$


## Repository Structure

```text
.
├── .github/              # GitHub issue and pull request templates
├── docs/                 # Project notes and documentation
├── examples/             # Example scripts and notebooks
├── src/                  # Source package code
├── tests/                # Unit and integration tests
├── README.md             # Project overview
├── pyproject.toml        # Python project metadata
└── requirements.txt      # Runtime dependencies
```

## Getting Started

### Preparing CONDA environments

Clone this repository:

```bash
git clone https://github.com/kezhoulumelody/Lorenz63_NeuralODE.git
cd Lorenz63_NeuralODE
```

Use the existing conda environment:

```bash
conda activate /data/kezhoulumelody/melody_NXRO
```

Install the core dependencies:

```bash
pip install -r requirements.txt
```

Install this project in editable mode:

```bash
pip install -e .
```

To deactivate the environment when finished:

```bash
conda deactivate
```

### Generating training dataset 

```bash
cd Lorenz63_NeuralODE/src/lorenz63_neuralode/ 
python data.py \
--output "Lorenz63_NeuralODE/data/lorenz63_trajectory_10-28-2.7.npz" \
--t-start 0 \
--t-end 1000 \
--dt 0.01 \
--sigma 10 \
--rho 28 \
--beta 2.7
```
Important flags:

- `--output`: output file is saved as a NumPy compressed data archive (".npz") format.
- `--t-start`: starting time.
- `--t-end`: end time.
- `--dt`: time interval.
- `--sigma`: parameter $\sigma$.
- `--rho`: parameter $\rho$.
- `--beta`, `parameter $\beta$.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for basic contribution guidelines.

## License

This project is currently provided under the MIT License. See [LICENSE](LICENSE) for details.
