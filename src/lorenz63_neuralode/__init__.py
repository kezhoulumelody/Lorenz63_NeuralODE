"""Utilities for Lorenz63 Neural ODE experiments."""

from .data import Lorenz63ResidualDataset, generate_lorenz63, get_dataloaders, lorenz63
from .models import (
    Lorenz63AttentionResidualModel,
    Lorenz63BilinearResidualModel,
    Lorenz63GraphResidualModel,
    Lorenz63LinearModel,
    Lorenz63PolynomialResidualModel,
    Lorenz63PureMLPModel,
    Lorenz63ResidualModel,
    Lorenz63ResidualMixModel,
    Lorenz63TransformerResidualModel,
    StructuredLorenz63Linear,
    build_lorenz63_model,
    lorenz63_linear_matrix,
    true_lorenz63_residual,
)

__version__ = "0.1.0"

__all__ = [
    "Lorenz63ResidualDataset",
    "Lorenz63AttentionResidualModel",
    "Lorenz63BilinearResidualModel",
    "Lorenz63GraphResidualModel",
    "Lorenz63LinearModel",
    "Lorenz63PolynomialResidualModel",
    "Lorenz63PureMLPModel",
    "Lorenz63ResidualModel",
    "Lorenz63ResidualMixModel",
    "Lorenz63TransformerResidualModel",
    "StructuredLorenz63Linear",
    "build_lorenz63_model",
    "generate_lorenz63",
    "get_dataloaders",
    "lorenz63",
    "lorenz63_linear_matrix",
    "true_lorenz63_residual",
]
