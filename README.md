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

src/lorenz63_neuralode/      
  models.py            #   Unstructured-linear residual models (MLP, attention, graph, transformer)
  models_structuredLinear.py #   Structured-linear model variants and baselines
  train.py             #   Training loops with val split support
  eval.py              #   Evaluation metrics
  data.py              #   Generating training dataset of Lorenz63 model. User could modify the length of the simulations, parameters, etc.,

examples/                   #   Jupyter notebook for running examples, diagnosing and plotting results
  plot_lorenz63_trajectory.ipynb     #   Visualize the training dataset.
  train_lorenz63_neural_residual.ipynb #   Train the unstructured-linear MLP residual model.
  predict_lorenz63_neural_residual_cookbook.ipynb #   Evaluate a pretrained unstructured-linear MLP residual model.
  train_lorenz63_stucturedLinear_model.ipynb #   Train structured-linear model variants.
  predict_lorenz63_pretrained_structuredLinear_cookbook.ipynb #   Evaluate structured-linear checkpoints.

data/                  # Training data
  lorenz63_trajectory_10-28-2.7.npz #   Example training dataset generated with  (10 climate indices, 1979-2024)

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

### Training

Use the unstructured-linear MLP residual training notebook:

```text
examples/train_lorenz63_neural_residual.ipynb
```

This notebook trains the current `models.py` MLP residual Neural ODE model:

$$
f_\theta(x) = A_\theta x + G_\theta(x)
$$

where `A_theta` is a full unconstrained `3 x 3` learnable matrix and the residual mask defaults to `(1, 1, 1)`, so the residual is not given the Lorenz63 nonlinear sparsity constraint. In the configuration cell, set:

```python
MODEL_NAME = "residual_mlp"
DATA_PATH = repo_root / "data" / "lorenz63_trajectory_10-28-2.7.npz"
OUTPUT_DIR = repo_root / "outputs" / "unstructured" / f"lorenz63_{MODEL_NAME}"
N_EPOCHS = 200
DEVICE = "cuda"                # use "cpu" if GPU is unavailable
```

Then run the notebook from top to bottom. The checkpoint is saved under `outputs/`, for example:

```text
outputs/unstructured/lorenz63_residual_mlp/lorenz63_residual_mlp_10-28-2.7_1-1-2.pt
```

The notebook also plots in-sample and out-of-sample RMSE and ACC curves during training.

For the structured-linear model family, use:

```text
examples/train_lorenz63_stucturedLinear_model.ipynb
```

### Evaluation

Use the unstructured-linear MLP residual prediction notebook:

```text
examples/predict_lorenz63_neural_residual_cookbook.ipynb
```

This notebook loads a pretrained unstructured-linear MLP residual checkpoint, runs trajectory prediction from a configurable initial condition, and compares the learned model against the reference Lorenz63 trajectory. In the configuration cell, set:

```python
DATA_PATH = repo_root / "data" / "lorenz63_trajectory_10-28-2.7.npz"
CHECKPOINT_PATH = repo_root / "outputs" / "unstructured" / "lorenz63_residual_mlp" / "lorenz63_residual_mlp_10-28-2.7_1-1-2.pt"
INITIAL_INDEX = 1000
ROLLOUT_STEPS = 4000
DEVICE = "cuda"
```

The notebook produces:

- true vs learned 3D trajectories,
- true vs learned `x`, `y`, and `z` time series,
- trajectory RMSE,
- learned unconstrained linear matrix comparison,
- learned residual comparison against the actual Lorenz63 residual,
- parity plots and summary statistics for states and tendencies.

The notebook checks that the checkpoint contains the current unstructured-linear key `linear.matrix_param`. If you want to evaluate a structured-linear checkpoint, use:

```text
examples/predict_lorenz63_pretrained_structuredLinear_cookbook.ipynb
```

If `models.py` has changed since the checkpoint was trained, restart the Jupyter kernel and retrain the model before evaluation.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for basic contribution guidelines.

## License

This project is currently provided under the MIT License. See [LICENSE](LICENSE) for details.
