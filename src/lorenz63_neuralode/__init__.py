"""Utilities for Lorenz63 Neural ODE experiments."""

from .data import Lorenz63ResidualDataset, generate_lorenz63, get_dataloaders, lorenz63
from .models import (
    Lorenz63AttentionResidualModel,
    Lorenz63GraphResidualModel,
    Lorenz63ResidualModel,
    Lorenz63TransformerResidualModel,
    Lorenz63UnstructuredAttentionResidualModel,
    Lorenz63UnstructuredGraphResidualModel,
    Lorenz63UnstructuredResidualModel,
    Lorenz63UnstructuredTransformerResidualModel,
    ResidualMLP,
    UnstructuredLorenz63Linear,
    build_lorenz63_model,
    lorenz63_linear_matrix,
    true_lorenz63_residual,
)
from .models_structuredLinear import (
    Lorenz63BilinearGraphResidualModel,
    Lorenz63BilinearResidualModel,
    Lorenz63GraphResidualModel as Lorenz63StructuredGraphResidualModel,
    Lorenz63LinearModel,
    Lorenz63PolynomialResidualModel,
    Lorenz63PureMLPModel,
    Lorenz63PureTransformerModel,
    Lorenz63ResidualMixModel,
    StructuredLorenz63Linear,
)

__version__ = "0.1.0"

__all__ = [
    "Lorenz63ResidualDataset",
    "Lorenz63AttentionResidualModel",
    "Lorenz63BilinearGraphResidualModel",
    "Lorenz63BilinearResidualModel",
    "Lorenz63GraphResidualModel",
    "Lorenz63LinearModel",
    "Lorenz63PolynomialResidualModel",
    "Lorenz63PureMLPModel",
    "Lorenz63PureTransformerModel",
    "Lorenz63ResidualModel",
    "Lorenz63ResidualMixModel",
    "Lorenz63StructuredGraphResidualModel",
    "Lorenz63TransformerResidualModel",
    "Lorenz63UnstructuredAttentionResidualModel",
    "Lorenz63UnstructuredGraphResidualModel",
    "Lorenz63UnstructuredResidualModel",
    "Lorenz63UnstructuredTransformerResidualModel",
    "ResidualMLP",
    "StructuredLorenz63Linear",
    "UnstructuredLorenz63Linear",
    "build_lorenz63_model",
    "generate_lorenz63",
    "get_dataloaders",
    "lorenz63",
    "lorenz63_linear_matrix",
    "true_lorenz63_residual",
]
