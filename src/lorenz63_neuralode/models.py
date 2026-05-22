"""Unstructured-linear Neural ODE models for Lorenz63 residual learning."""

from __future__ import annotations

import torch
import torch.nn as nn


def lorenz63_linear_matrix(
    sigma: float = 10.0,
    rho: float = 28.0,
    beta: float = 8.0 / 3.0,
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return the standard Lorenz63 linear operator used for initialization."""
    return torch.tensor(
        [
            [-sigma, sigma, 0.0],
            [rho, -1.0, 0.0],
            [0.0, 0.0, -beta],
        ],
        dtype=dtype,
    )


class UnstructuredLorenz63Linear(nn.Module):
    """Learnable 3x3 linear operator without Lorenz63 structural constraints.

    The default initialization starts from the Lorenz63 linear matrix, but each
    matrix entry is learned independently during training.
    """

    def __init__(
        self,
        sigma_init: float = 10.0,
        rho_init: float = 28.0,
        beta_init: float = 8.0 / 3.0,
        matrix_init: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if matrix_init is None:
            matrix_init = lorenz63_linear_matrix(
                sigma=sigma_init,
                rho=rho_init,
                beta=beta_init,
            )
        if matrix_init.shape != (3, 3):
            raise ValueError(f"matrix_init must have shape (3, 3), got {tuple(matrix_init.shape)}")
        self.matrix_param = nn.Parameter(matrix_init.detach().clone().to(dtype=torch.float32))

    @property
    def matrix(self) -> torch.Tensor:
        """Return the current unconstrained learnable matrix A_theta."""
        return self.matrix_param

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute A_theta x for a state batch shaped [..., 3]."""
        return torch.matmul(x, self.matrix.T)


def true_lorenz63_residual(x: torch.Tensor) -> torch.Tensor:
    """Return the exact nonlinear Lorenz63 residual [0, -xz, xy]."""
    residual = torch.zeros_like(x)
    residual[..., 1] = -x[..., 0] * x[..., 2]
    residual[..., 2] = x[..., 0] * x[..., 1]
    return residual


class ResidualMLP(nn.Module):
    """Small MLP used to learn the nonlinear residual."""

    def __init__(
        self,
        input_dim: int = 3,
        hidden_dim: int = 64,
        output_dim: int = 3,
        n_hidden_layers: int = 2,
        activation: type[nn.Module] = nn.Tanh,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        dims = [input_dim] + [hidden_dim] * n_hidden_layers + [output_dim]

        for in_dim, out_dim in zip(dims[:-2], dims[1:-1], strict=True):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(activation())
        layers.append(nn.Linear(dims[-2], dims[-1]))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Lorenz63ResidualModel(nn.Module):
    """Unconstrained linear dynamics plus MLP nonlinear residual.

    The model computes the explicit split:

        f_theta(x) = A_theta x + G_theta(x)

    where ``A_theta`` is a full learnable 3x3 matrix and ``G_theta`` is a
    nonlinear residual network.
    """

    def __init__(
        self,
        sigma_init: float = 10.0,
        rho_init: float = 28.0,
        beta_init: float = 8.0 / 3.0,
        hidden_dim: int = 64,
        n_hidden_layers: int = 2,
        residual_mask: tuple[float, float, float] = (1.0, 1.0, 1.0),
        matrix_init: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.linear = UnstructuredLorenz63Linear(
            sigma_init=sigma_init,
            rho_init=rho_init,
            beta_init=beta_init,
            matrix_init=matrix_init,
        )
        self.register_buffer("residual_mask", torch.tensor(residual_mask, dtype=torch.float32))
        self.residual_net = ResidualMLP(
            input_dim=3,
            hidden_dim=hidden_dim,
            output_dim=3,
            n_hidden_layers=n_hidden_layers,
        )

    def linear_tendency(self, x: torch.Tensor) -> torch.Tensor:
        """Compute A_theta x for a state batch shaped [..., 3]."""
        return self.linear(x)

    @property
    def A(self) -> torch.Tensor:
        """Return the current learned linear matrix."""
        return self.linear.matrix

    def residual_tendency(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the learned nonlinear residual G_theta(x)."""
        return self.residual_net(x) * self.residual_mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return dx/dt for state x with shape [..., 3]."""
        return self.linear_tendency(x) + self.residual_tendency(x)


class Lorenz63AttentionResidualModel(nn.Module):
    """Unconstrained linear dynamics plus variable-token attention residual."""

    def __init__(
        self,
        sigma_init: float = 10.0,
        rho_init: float = 28.0,
        beta_init: float = 8.0 / 3.0,
        d_model: int = 32,
        dropout: float = 0.0,
        mask_mode: str = "full",
        residual_mask: tuple[float, float, float] = (1.0, 1.0, 1.0),
        matrix_init: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.linear = UnstructuredLorenz63Linear(
            sigma_init=sigma_init,
            rho_init=rho_init,
            beta_init=beta_init,
            matrix_init=matrix_init,
        )
        self.register_buffer("residual_mask", torch.tensor(residual_mask, dtype=torch.float32))
        self.W_input = nn.Linear(1, d_model)
        self.var_embedding = nn.Parameter(torch.randn(3, d_model) * 0.02)
        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        self.Wo = nn.Linear(d_model, 1, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        if mask_mode == "lorenz":
            mask = torch.tensor(
                [
                    [1.0, 1.0, 0.0],
                    [1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0],
                ],
                dtype=torch.float32,
            )
        elif mask_mode == "full":
            mask = torch.ones(3, 3)
        else:
            raise ValueError("mask_mode must be 'lorenz' or 'full'")
        self.register_buffer("attn_mask", mask)

    def linear_tendency(self, x: torch.Tensor) -> torch.Tensor:
        """Compute A_theta x for a state batch shaped [..., 3]."""
        return self.linear(x)

    @property
    def A(self) -> torch.Tensor:
        """Return the current learned linear matrix."""
        return self.linear.matrix

    def residual_tendency(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the learned nonlinear residual G_theta(x)."""
        tokens = self.W_input(x.unsqueeze(-1)) + self.var_embedding.unsqueeze(0)
        q = self.Wq(tokens)
        k = self.Wk(tokens)
        v = self.Wv(tokens)
        scores = torch.einsum("bid,bjd->bij", q, k) / (self.d_model**0.5)
        scores = scores.masked_fill(self.attn_mask.unsqueeze(0) < 0.5, -1e9)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.einsum("bij,bjd->bid", attn, v)
        return self.Wo(out).squeeze(-1) * self.residual_mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return dx/dt for state x with shape [..., 3]."""
        return self.linear_tendency(x) + self.residual_tendency(x)


class Lorenz63GraphResidualModel(nn.Module):
    """Unconstrained linear dynamics plus graph-message nonlinear residual."""

    def __init__(
        self,
        sigma_init: float = 10.0,
        rho_init: float = 28.0,
        beta_init: float = 8.0 / 3.0,
        use_fixed_graph: bool = True,
        residual_mask: tuple[float, float, float] = (1.0, 1.0, 1.0),
        matrix_init: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.use_fixed_graph = use_fixed_graph
        self.linear = UnstructuredLorenz63Linear(
            sigma_init=sigma_init,
            rho_init=rho_init,
            beta_init=beta_init,
            matrix_init=matrix_init,
        )
        self.register_buffer("residual_mask", torch.tensor(residual_mask, dtype=torch.float32))
        lorenz_adj = torch.tensor(
            [
                [1.0, 1.0, 0.0],
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
            ],
            dtype=torch.float32,
        )
        if use_fixed_graph:
            self.register_buffer("A_graph", lorenz_adj)
        else:
            self.A_graph_param = nn.Parameter(lorenz_adj)
        self.W_g = nn.Parameter(torch.empty(3, 3))
        nn.init.xavier_uniform_(self.W_g)

    def linear_tendency(self, x: torch.Tensor) -> torch.Tensor:
        """Compute A_theta x for a state batch shaped [..., 3]."""
        return self.linear(x)

    @property
    def A(self) -> torch.Tensor:
        """Return the current learned linear matrix."""
        return self.linear.matrix

    def normalized_graph(self) -> torch.Tensor:
        """Return row-normalized graph weights for residual message passing."""
        graph = self.A_graph if self.use_fixed_graph else torch.relu(self.A_graph_param)
        rowsum = graph.sum(dim=1, keepdim=True).clamp_min(1e-6)
        return graph / rowsum

    def residual_tendency(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the learned graph-message residual G_theta(x)."""
        message = torch.matmul(x, self.normalized_graph().T)
        return torch.tanh(torch.matmul(message, self.W_g.T)) * self.residual_mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return dx/dt for state x with shape [..., 3]."""
        return self.linear_tendency(x) + self.residual_tendency(x)


class Lorenz63TransformerResidualModel(nn.Module):
    """Unconstrained linear dynamics plus transformer-encoder residual."""

    def __init__(
        self,
        sigma_init: float = 10.0,
        rho_init: float = 28.0,
        beta_init: float = 8.0 / 3.0,
        d_model: int = 32,
        n_heads: int = 4,
        n_layers: int = 1,
        dropout: float = 0.0,
        residual_mask: tuple[float, float, float] = (1.0, 1.0, 1.0),
        matrix_init: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.linear = UnstructuredLorenz63Linear(
            sigma_init=sigma_init,
            rho_init=rho_init,
            beta_init=beta_init,
            matrix_init=matrix_init,
        )
        self.register_buffer("residual_mask", torch.tensor(residual_mask, dtype=torch.float32))
        self.input_proj = nn.Linear(1, d_model)
        self.var_embedding = nn.Parameter(torch.randn(3, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=2 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output_proj = nn.Linear(d_model, 1)

    def linear_tendency(self, x: torch.Tensor) -> torch.Tensor:
        """Compute A_theta x for a state batch shaped [..., 3]."""
        return self.linear(x)

    @property
    def A(self) -> torch.Tensor:
        """Return the current learned linear matrix."""
        return self.linear.matrix

    def residual_tendency(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the learned nonlinear residual G_theta(x)."""
        tokens = self.input_proj(x.unsqueeze(-1)) + self.var_embedding.unsqueeze(0)
        encoded = self.encoder(tokens)
        return self.output_proj(encoded).squeeze(-1) * self.residual_mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return dx/dt for state x with shape [..., 3]."""
        return self.linear_tendency(x) + self.residual_tendency(x)


Lorenz63UnstructuredResidualModel = Lorenz63ResidualModel
Lorenz63UnstructuredAttentionResidualModel = Lorenz63AttentionResidualModel
Lorenz63UnstructuredGraphResidualModel = Lorenz63GraphResidualModel
Lorenz63UnstructuredTransformerResidualModel = Lorenz63TransformerResidualModel


def build_lorenz63_model(
    model_name: str = "residual_mlp",
    sigma_init: float = 10.0,
    rho_init: float = 28.0,
    beta_init: float = 8.0 / 3.0,
    hidden_dim: int = 64,
    n_hidden_layers: int = 2,
) -> nn.Module:
    """Build an unstructured-linear Lorenz63 residual model by name."""
    model_name = model_name.lower()
    if model_name in {
        "residual_mlp",
        "mlp_residual",
        "neural_ode_residual",
        "unstructured_residual_mlp",
        "unstructured_mlp",
        "free_linear_mlp",
    }:
        return Lorenz63ResidualModel(
            sigma_init=sigma_init,
            rho_init=rho_init,
            beta_init=beta_init,
            hidden_dim=hidden_dim,
            n_hidden_layers=n_hidden_layers,
        )
    if model_name in {"attention", "attentive", "unstructured_attention", "free_linear_attention"}:
        return Lorenz63AttentionResidualModel(
            sigma_init=sigma_init,
            rho_init=rho_init,
            beta_init=beta_init,
            d_model=hidden_dim,
        )
    if model_name in {"graph", "graph_residual", "unstructured_graph", "free_linear_graph"}:
        return Lorenz63GraphResidualModel(
            sigma_init=sigma_init,
            rho_init=rho_init,
            beta_init=beta_init,
        )
    if model_name in {
        "transformer",
        "transformer_residual",
        "unstructured_transformer",
        "free_linear_transformer",
    }:
        return Lorenz63TransformerResidualModel(
            sigma_init=sigma_init,
            rho_init=rho_init,
            beta_init=beta_init,
            d_model=hidden_dim,
            n_layers=n_hidden_layers,
        )
    raise ValueError(
        f"Unknown model_name={model_name!r}. Choose one of: "
        "residual_mlp, attention, graph, transformer."
    )
