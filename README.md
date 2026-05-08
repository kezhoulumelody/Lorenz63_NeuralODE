# Lorenz63 Neural ODE

Neural ordinary differential equation experiments for learning, simulating, and analyzing the Lorenz63 dynamical system.

## Overview

The Lorenz63 system is a classic chaotic dynamical system defined by three coupled ordinary differential equations. This repository is intended to provide a clean workspace for experiments that combine the Lorenz63 model with Neural ODE methods.

Potential project goals include:

- Generate Lorenz63 trajectories from known parameters.
- Train Neural ODE models to approximate Lorenz63 dynamics.
- Compare learned trajectories against numerical solvers.
- Visualize attractors, prediction error, and long-horizon behavior.

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

### Quick Start

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

## Usage

Example entry points and notebooks can be added under `examples/`.

Suggested workflow:

1. Add Lorenz63 simulation utilities under `src/lorenz63_neuralode/`.
2. Add model training code under `src/lorenz63_neuralode/`.
3. Add runnable scripts or notebooks under `examples/`.
4. Add tests under `tests/`.

## Development

Install development dependencies once they are added to `pyproject.toml` or a dedicated development requirements file.

Run tests:

```bash
pytest
```

Format and lint commands can be added here when the project selects tools such as `ruff`, `black`, or `mypy`.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for basic contribution guidelines.

## License

This project is currently provided under the MIT License. See [LICENSE](LICENSE) for details.
