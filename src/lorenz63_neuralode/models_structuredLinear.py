"""Neural ODE models for Lorenz63 residual learning."""

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
    """Return the known Lorenz63 linear operator A.

    With x = [x, y, z]^T, the split is:

        dx/dt = A x + G(x)

    where G(x) = [0, -xz, xy]^T.
    """
    return torch.tensor(
        [
            [-sigma, sigma, 0.0],
            [rho, -1.0, 0.0],
            [0.0, 0.0, -beta],
        ],
        dtype=dtype,
    )


class StructuredLorenz63Linear(nn.Module):
    """Learnable Lorenz63 linear operator with fixed sparsity pattern.

    The learnable matrix is constrained to the Lorenz63 linear form:

        [[-sigma, sigma, 0],
         [rho,    -1,    0],
         [0,       0,   -beta]]

    ``sigma``, ``rho``, and ``beta`` are learned directly, while the signs and
    fixed y damping term are built into the matrix.
    """

    def __init__(
        self,
        sigma_init: float = 10.0,
        rho_init: float = 28.0,
        beta_init: float = 8.0 / 3.0,
    ) -> None:
        super().__init__()
        init_values = torch.tensor([sigma_init, rho_init, beta_init], dtype=torch.float32)
        self.parameters_lorenz = nn.Parameter(init_values)

    @property
    def physical_parameters(self) -> torch.Tensor:
        """Return learned parameters [sigma, rho, beta]."""
        return self.parameters_lorenz

    @property
    def matrix(self) -> torch.Tensor:
        """Return the structured learnable matrix A_theta."""
        sigma, rho, beta = self.physical_parameters
        zero = sigma.new_tensor(0.0)
        minus_one = sigma.new_tensor(-1.0)
        return torch.stack(
            [
                torch.stack([-sigma, sigma, zero]),
                torch.stack([rho, minus_one, zero]),
                torch.stack([zero, zero, -beta]),
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute A_theta x for a batch shaped [..., 3]."""
        return torch.matmul(x, self.matrix.T)


class UnstructuredLorenz63Linear(nn.Module):
    """Learnable 3x3 linear operator without Lorenz63 structural constraints.

    This keeps the explicit split ``f_theta(x) = A_theta x + G_theta(x)``, but
    every entry of ``A_theta`` is learned independently instead of being tied to
    the Lorenz63 linear form.
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
        """Return the unconstrained learnable matrix A_theta."""
        return self.matrix_param

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute A_theta x for a batch shaped [..., 3]."""
        return torch.matmul(x, self.matrix.T)


def true_lorenz63_residual(x: torch.Tensor) -> torch.Tensor:
    """For sanity checks: return the exact nonlinear Lorenz63 residual [0, -xz, xy]."""
    residual = torch.zeros_like(x)
    residual[..., 1] = -x[..., 0] * x[..., 2]
    residual[..., 2] = x[..., 0] * x[..., 1]
    return residual


class ResidualMLP(nn.Module):
    """Small MLP used to learn the nonlinear Lorenz63 residual."""

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
    """Structured learnable linear dynamics plus learned nonlinear residual.

    The model computes:

        f_theta(x) = A_theta x + G_theta(x)

    The default residual mask forces the first residual component to zero,
    matching the Lorenz63 split G(x) = [0, -xz, xy]^T.
    """

    def __init__(
        self,
        sigma_init: float = 10.0,
        rho_init: float = 28.0,
        beta_init: float = 8.0 / 3.0,
        hidden_dim: int = 64,
        n_hidden_layers: int = 2,
        residual_mask: tuple[float, float, float] = (0.0, 1.0, 1.0), ## Force the first residual component to be zero, matching the Lorenz63 split G(x) = [0, -xz, xy]^T.
    ) -> None:
        super().__init__()
        self.linear = StructuredLorenz63Linear(
            sigma_init=sigma_init,
            rho_init=rho_init,
            beta_init=beta_init,
        )
        self.register_buffer("residual_mask", torch.tensor(residual_mask, dtype=torch.float32))
        self.residual_net = ResidualMLP(
            input_dim=3,
            hidden_dim=hidden_dim,
            output_dim=3,
            n_hidden_layers=n_hidden_layers,
        )

    def linear_tendency(self, x: torch.Tensor) -> torch.Tensor:
        """Compute A_theta x for a batch shaped [..., 3]."""
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


class Lorenz63UnstructuredResidualModel(nn.Module):
    """Unconstrained learnable linear dynamics plus learned nonlinear residual.

    The model still computes an explicit additive decomposition:

        f_theta(x) = A_theta x + G_theta(x)

    Unlike ``Lorenz63ResidualModel``, ``A_theta`` is a full learnable 3x3
    matrix. The residual MLP therefore knows it is modeling the component left
    after an independently learned linear tendency.
    """

    def __init__(
        self,
        sigma_init: float = 10.0,
        rho_init: float = 28.0,
        beta_init: float = 8.0 / 3.0,
        hidden_dim: int = 64,
        n_hidden_layers: int = 2,
        residual_mask: tuple[float, float, float] = (0.0, 1.0, 1.0),
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
        """Compute A_theta x for a batch shaped [..., 3]."""
        return self.linear(x)

    @property
    def A(self) -> torch.Tensor:
        """Return the current learned unconstrained linear matrix."""
        return self.linear.matrix

    def residual_tendency(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the learned nonlinear residual G_theta(x)."""
        return self.residual_net(x) * self.residual_mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return dx/dt for state x with shape [..., 3]."""
        return self.linear_tendency(x) + self.residual_tendency(x)


class Lorenz63LinearModel(nn.Module):
    """Linear baseline model: structured learnable Lorenz63 linear model: f_theta(x) = A_theta x."""

    def __init__(
        self,
        sigma_init: float = 10.0,
        rho_init: float = 28.0,
        beta_init: float = 8.0 / 3.0,
    ) -> None:
        super().__init__()
        self.linear = StructuredLorenz63Linear(
            sigma_init=sigma_init,
            rho_init=rho_init,
            beta_init=beta_init,
        )

    @property
    def A(self) -> torch.Tensor:
        return self.linear.matrix

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class Lorenz63PolynomialResidualModel(nn.Module):
    """Structured linear model plus learned polynomial residual basis.

    The residual basis is ``[x^2, y^2, z^2, xy, xz, yz]``. This is useful for
    Lorenz63 because the true nonlinear terms are exactly quadratic.
    """

    def __init__(
        self,
        sigma_init: float = 10.0,
        rho_init: float = 28.0,
        beta_init: float = 8.0 / 3.0,
        residual_mask: tuple[float, float, float] = (0.0, 1.0, 1.0),
    ) -> None:
        super().__init__()
        self.linear = StructuredLorenz63Linear(
            sigma_init=sigma_init,
            rho_init=rho_init,
            beta_init=beta_init,
        )
        self.register_buffer("residual_mask", torch.tensor(residual_mask, dtype=torch.float32))
        self.W_poly = nn.Parameter(torch.zeros(6, 3))
        nn.init.xavier_uniform_(self.W_poly)

    @property
    def A(self) -> torch.Tensor:
        return self.linear.matrix

    def polynomial_features(self, x: torch.Tensor) -> torch.Tensor:
        x0, x1, x2 = x[..., 0], x[..., 1], x[..., 2]
        return torch.stack(
            [
                x0 * x0,
                x1 * x1,
                x2 * x2,
                x0 * x1,
                x0 * x2,
                x1 * x2,
            ],
            dim=-1,
        )

    def residual_tendency(self, x: torch.Tensor) -> torch.Tensor:
        return torch.matmul(self.polynomial_features(x), self.W_poly) * self.residual_mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) + self.residual_tendency(x)


class Lorenz63ResidualMixModel(Lorenz63ResidualModel):
    """Structured linear plus scaled residual MLP: A_theta x + alpha R_theta(x). Similar to Lorenz63ResidualModel but controls how strong the residual contribution is"""

    def __init__(
        self,
        sigma_init: float = 10.0,
        rho_init: float = 28.0,
        beta_init: float = 8.0 / 3.0,
        hidden_dim: int = 64,
        n_hidden_layers: int = 2,
        alpha_init: float = 0.1,
        alpha_learnable: bool = False,
        residual_mask: tuple[float, float, float] = (0.0, 1.0, 1.0),
    ) -> None:
        super().__init__(
            sigma_init=sigma_init,
            rho_init=rho_init,
            beta_init=beta_init,
            hidden_dim=hidden_dim,
            n_hidden_layers=n_hidden_layers,
            residual_mask=residual_mask,
        )
        if alpha_learnable:
            self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        else:
            self.register_buffer("alpha", torch.tensor(float(alpha_init)))

    def residual_tendency(self, x: torch.Tensor) -> torch.Tensor:
        return self.alpha * super().residual_tendency(x)


class Lorenz63PureMLPModel(nn.Module):
    """Pure neural ODE baseline: an MLP learns the full tendency directly."""

    def __init__(
        self,
        hidden_dim: int = 64,
        n_hidden_layers: int = 2,
    ) -> None:
        super().__init__()
        self.drift = ResidualMLP(
            input_dim=3,
            hidden_dim=hidden_dim,
            output_dim=3,
            n_hidden_layers=n_hidden_layers,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drift(x)


class Lorenz63PureTransformerModel(nn.Module):
    """Pure neural ODE baseline: a transformer learns the full tendency directly.

    Unlike ``Lorenz63TransformerResidualModel``, this model does not use the
    structured Lorenz63 linear operator or a residual mask. It predicts the
    complete tendency ``[dx/dt, dy/dt, dz/dt]`` from variable tokens.
    """

    def __init__(
        self,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != 3:
            raise ValueError(f"Expected final state dimension 3, got shape {tuple(x.shape)}")
        leading_shape = x.shape[:-1]
        flat_x = x.reshape(-1, 3)
        tokens = self.input_proj(flat_x.unsqueeze(-1)) + self.var_embedding.unsqueeze(0)
        encoded = self.encoder(tokens)
        return self.output_proj(encoded).squeeze(-1).reshape(*leading_shape, 3)


class Lorenz63BilinearResidualModel(nn.Module):
    """Structured linear plus low-rank bilinear residual channels."""

    def __init__(
        self,
        sigma_init: float = 10.0,
        rho_init: float = 28.0,
        beta_init: float = 8.0 / 3.0,
        n_channels: int = 4,
        rank: int = 2,
        residual_mask: tuple[float, float, float] = (0.0, 1.0, 1.0),
    ) -> None:
        super().__init__()
        self.linear = StructuredLorenz63Linear(
            sigma_init=sigma_init,
            rho_init=rho_init,
            beta_init=beta_init,
        )
        self.register_buffer("residual_mask", torch.tensor(residual_mask, dtype=torch.float32))
        self.P = nn.Parameter(torch.empty(n_channels, 3, rank))
        self.Q = nn.Parameter(torch.empty(n_channels, 3, rank))
        self.W_proj = nn.Parameter(torch.empty(n_channels, 3))
        nn.init.xavier_uniform_(self.P)
        nn.init.xavier_uniform_(self.Q)
        nn.init.xavier_uniform_(self.W_proj)

    @property
    def A(self) -> torch.Tensor:
        return self.linear.matrix

    def residual_tendency(self, x: torch.Tensor) -> torch.Tensor:
        p_feat = torch.einsum("bn,cnr->bcr", x, self.P)
        q_feat = torch.einsum("bn,cnr->bcr", x, self.Q)
        channels = torch.sum(p_feat * q_feat, dim=-1)
        return torch.matmul(channels, self.W_proj) * self.residual_mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) + self.residual_tendency(x)


class Lorenz63BilinearGraphResidualModel(nn.Module):
    """Structured linear plus explicit graph-masked bilinear residual.

    The residual has the form

        r_i(x) = sum_{j,k} A_ijk B_ijk x_j x_k,

    where ``A_ijk`` gates which variable-pair products contribute to each
    output tendency and ``B_ijk`` is learned.
    """

    def __init__(
        self,
        sigma_init: float = 10.0,
        rho_init: float = 28.0,
        beta_init: float = 8.0 / 3.0,
        use_fixed_graph: bool = True,
        mask_mode: str = "lorenz",
        residual_mask: tuple[float, float, float] = (0.0, 1.0, 1.0),
    ) -> None:
        super().__init__()
        self.use_fixed_graph = use_fixed_graph
        self.linear = StructuredLorenz63Linear(
            sigma_init=sigma_init,
            rho_init=rho_init,
            beta_init=beta_init,
        )
        self.register_buffer("residual_mask", torch.tensor(residual_mask, dtype=torch.float32))

        if mask_mode == "lorenz":
            pair_graph = torch.zeros(3, 3, 3, dtype=torch.float32)
            pair_graph[1, 0, 2] = 1.0  # x*z contributes to dy/dt.
            pair_graph[2, 0, 1] = 1.0  # x*y contributes to dz/dt.
        elif mask_mode == "full":
            pair_graph = torch.ones(3, 3, 3, dtype=torch.float32)
        else:
            raise ValueError("mask_mode must be 'lorenz' or 'full'")

        if use_fixed_graph:
            self.register_buffer("pair_graph", pair_graph)
        else:
            self.pair_graph_param = nn.Parameter(pair_graph)

        self.B_pair = nn.Parameter(torch.empty(3, 3, 3))
        nn.init.xavier_uniform_(self.B_pair)

    @property
    def A(self) -> torch.Tensor:
        return self.linear.matrix

    def graph_mask(self) -> torch.Tensor:
        if self.use_fixed_graph:
            return self.pair_graph
        return torch.relu(self.pair_graph_param)

    @property
    def bilinear_weights(self) -> torch.Tensor:
        """Return the effective A_ijk B_ijk bilinear residual weights."""
        return self.graph_mask() * self.B_pair

    def residual_tendency(self, x: torch.Tensor) -> torch.Tensor:
        pair_products = x.unsqueeze(2) * x.unsqueeze(1)
        residual = torch.einsum("bjk,ijk->bi", pair_products, self.bilinear_weights)
        return residual * self.residual_mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) + self.residual_tendency(x)


class Lorenz63AttentionResidualModel(nn.Module):
    """Structured linear plus lightweight variable-token attention residual."""

    def __init__(
        self,
        sigma_init: float = 10.0,
        rho_init: float = 28.0,
        beta_init: float = 8.0 / 3.0,
        d_model: int = 32,
        dropout: float = 0.0,
        mask_mode: str = "full",
        residual_mask: tuple[float, float, float] = (0.0, 1.0, 1.0),
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.linear = StructuredLorenz63Linear(
            sigma_init=sigma_init,
            rho_init=rho_init,
            beta_init=beta_init,
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
                ]
            )
        else:
            mask = torch.ones(3, 3)
        self.register_buffer("attn_mask", mask)

    @property
    def A(self) -> torch.Tensor:
        return self.linear.matrix

    def residual_tendency(self, x: torch.Tensor) -> torch.Tensor:
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
        return self.linear(x) + self.residual_tendency(x)


class Lorenz63UnstructuredAttentionResidualModel(Lorenz63AttentionResidualModel):
    """Unconstrained linear dynamics plus attention-based nonlinear residual."""

    def __init__(
        self,
        sigma_init: float = 10.0,
        rho_init: float = 28.0,
        beta_init: float = 8.0 / 3.0,
        d_model: int = 32,
        dropout: float = 0.0,
        mask_mode: str = "full",
        residual_mask: tuple[float, float, float] = (0.0, 1.0, 1.0),
        matrix_init: torch.Tensor | None = None,
    ) -> None:
        super().__init__(
            sigma_init=sigma_init,
            rho_init=rho_init,
            beta_init=beta_init,
            d_model=d_model,
            dropout=dropout,
            mask_mode=mask_mode,
            residual_mask=residual_mask,
        )
        self.linear = UnstructuredLorenz63Linear(
            sigma_init=sigma_init,
            rho_init=rho_init,
            beta_init=beta_init,
            matrix_init=matrix_init,
        )


class Lorenz63GraphResidualModel(nn.Module):
    """Structured linear plus graph-message residual over x, y, z variables."""

    def __init__(
        self,
        sigma_init: float = 10.0,
        rho_init: float = 28.0,
        beta_init: float = 8.0 / 3.0,
        use_fixed_graph: bool = True,
        residual_mask: tuple[float, float, float] = (0.0, 1.0, 1.0),
    ) -> None:
        super().__init__()
        self.use_fixed_graph = use_fixed_graph
        self.linear = StructuredLorenz63Linear(
            sigma_init=sigma_init,
            rho_init=rho_init,
            beta_init=beta_init,
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
            self.register_buffer("A_graph", lorenz_adj) # meaning that adjancency matris is set as lorenz_adj and keeps it fixed
        else:
            self.A_graph_param = nn.Parameter(lorenz_adj) # initial lorenz_adj but learnable during training
        self.W_g = nn.Parameter(torch.empty(3, 3))
        nn.init.xavier_uniform_(self.W_g) # Initialize the weights of the graph message function with Xavier uniform initialization, which is a common choice for linear layers.

    @property
    def A(self) -> torch.Tensor:
        return self.linear.matrix

    def normalized_graph(self) -> torch.Tensor:
        graph = self.A_graph if self.use_fixed_graph else torch.relu(self.A_graph_param) # relu forceses the graph weights to be non-negative, which can help with stability and interpretability in a graph-based model. It ensures that the learned graph structure does not have negative edge weights, which might not make sense in many contexts. By applying ReLU, we can encourage the model to learn a sparse and interpretable graph structure where edges represent positive relationships between variables. If the original A_graph_param has negative values, they will be set to zero, effectively removing those edges from the graph. This can lead to a more stable training process and a more meaningful learned graph structure. If use_fixed_graph is True, then A_graph is already non-negative (since it's initialized as lorenz_adj), so we can skip the ReLU in that case.
        rowsum = graph.sum(dim=1, keepdim=True).clamp_min(1e-6)
        return graph / rowsum

    def residual_tendency(self, x: torch.Tensor) -> torch.Tensor:
        message = torch.matmul(x, self.normalized_graph().T)
        return torch.tanh(torch.matmul(message, self.W_g.T)) * self.residual_mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) + self.residual_tendency(x)


class Lorenz63UnstructuredGraphResidualModel(Lorenz63GraphResidualModel):
    """Unconstrained linear dynamics plus graph-message nonlinear residual."""

    def __init__(
        self,
        sigma_init: float = 10.0,
        rho_init: float = 28.0,
        beta_init: float = 8.0 / 3.0,
        use_fixed_graph: bool = True,
        residual_mask: tuple[float, float, float] = (0.0, 1.0, 1.0),
        matrix_init: torch.Tensor | None = None,
    ) -> None:
        super().__init__(
            sigma_init=sigma_init,
            rho_init=rho_init,
            beta_init=beta_init,
            use_fixed_graph=use_fixed_graph,
            residual_mask=residual_mask,
        )
        self.linear = UnstructuredLorenz63Linear(
            sigma_init=sigma_init,
            rho_init=rho_init,
            beta_init=beta_init,
            matrix_init=matrix_init,
        )


class Lorenz63TransformerResidualModel(nn.Module):
    """Structured linear plus tiny transformer encoder over variable tokens."""

    def __init__(
        self,
        sigma_init: float = 10.0,
        rho_init: float = 28.0,
        beta_init: float = 8.0 / 3.0,
        d_model: int = 32,
        n_heads: int = 4,
        n_layers: int = 1,
        dropout: float = 0.0,
        residual_mask: tuple[float, float, float] = (0.0, 1.0, 1.0),
    ) -> None:
        super().__init__()
        self.linear = StructuredLorenz63Linear(
            sigma_init=sigma_init,
            rho_init=rho_init,
            beta_init=beta_init,
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

    @property
    def A(self) -> torch.Tensor:
        return self.linear.matrix

    def residual_tendency(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.input_proj(x.unsqueeze(-1)) + self.var_embedding.unsqueeze(0)
        encoded = self.encoder(tokens)
        return self.output_proj(encoded).squeeze(-1) * self.residual_mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) + self.residual_tendency(x)


def build_lorenz63_model(
    model_name: str = "residual_mlp",
    sigma_init: float = 10.0,
    rho_init: float = 28.0,
    beta_init: float = 8.0 / 3.0,
    hidden_dim: int = 64,
    n_hidden_layers: int = 2,
) -> nn.Module:
    """Build a Lorenz63 model by name."""
    model_name = model_name.lower()
    if model_name == "linear":
        return Lorenz63LinearModel(sigma_init=sigma_init, rho_init=rho_init, beta_init=beta_init)
    if model_name in {"polynomial", "poly"}:
        return Lorenz63PolynomialResidualModel(
            sigma_init=sigma_init,
            rho_init=rho_init,
            beta_init=beta_init,
        )
    if model_name in {"residual_mlp", "mlp_residual", "structured_mlp"}:
        return Lorenz63ResidualModel(
            sigma_init=sigma_init,
            rho_init=rho_init,
            beta_init=beta_init,
            hidden_dim=hidden_dim,
            n_hidden_layers=n_hidden_layers,
        )
    if model_name in {"unstructured_residual_mlp", "unstructured_mlp", "free_linear_mlp"}:
        return Lorenz63UnstructuredResidualModel(
            sigma_init=sigma_init,
            rho_init=rho_init,
            beta_init=beta_init,
            hidden_dim=hidden_dim,
            n_hidden_layers=n_hidden_layers,
        )
    if model_name in {"residual_mix", "mix"}:
        return Lorenz63ResidualMixModel(
            sigma_init=sigma_init,
            rho_init=rho_init,
            beta_init=beta_init,
            hidden_dim=hidden_dim,
            n_hidden_layers=n_hidden_layers,
        )
    if model_name in {"pure_mlp", "pure"}:
        return Lorenz63PureMLPModel(hidden_dim=hidden_dim, n_hidden_layers=n_hidden_layers)
    if model_name in {"pure_transformer", "transformer_pure", "full_transformer"}:
        return Lorenz63PureTransformerModel(
            d_model=hidden_dim,
            n_layers=n_hidden_layers,
        )
    if model_name in {"bilinear", "bilinear_residual"}:
        return Lorenz63BilinearResidualModel(
            sigma_init=sigma_init,
            rho_init=rho_init,
            beta_init=beta_init,
        )
    if model_name in {"bilinear_graph", "bilinear_graph_residual", "graph_bilinear"}:
        return Lorenz63BilinearGraphResidualModel(
            sigma_init=sigma_init,
            rho_init=rho_init,
            beta_init=beta_init,
        )
    if model_name in {"attention", "attentive"}:
        return Lorenz63AttentionResidualModel(
            sigma_init=sigma_init,
            rho_init=rho_init,
            beta_init=beta_init,
            d_model=hidden_dim,
        )
    if model_name in {"unstructured_attention", "free_linear_attention"}:
        return Lorenz63UnstructuredAttentionResidualModel(
            sigma_init=sigma_init,
            rho_init=rho_init,
            beta_init=beta_init,
            d_model=hidden_dim,
        )
    if model_name in {"graph", "graph_residual"}:
        return Lorenz63GraphResidualModel(
            sigma_init=sigma_init,
            rho_init=rho_init,
            beta_init=beta_init,
        )
    if model_name in {"unstructured_graph", "unstructured_graph_residual", "free_linear_graph"}:
        return Lorenz63UnstructuredGraphResidualModel(
            sigma_init=sigma_init,
            rho_init=rho_init,
            beta_init=beta_init,
        )
    if model_name in {"transformer", "transformer_residual"}:
        return Lorenz63TransformerResidualModel(
            sigma_init=sigma_init,
            rho_init=rho_init,
            beta_init=beta_init,
            d_model=hidden_dim,
        )
    raise ValueError(
        f"Unknown model_name={model_name!r}. Choose one of: "
        "linear, polynomial, residual_mlp, unstructured_residual_mlp, residual_mix, pure_mlp, "
        "pure_transformer, bilinear, bilinear_graph, attention, unstructured_attention, "
        "graph, unstructured_graph, transformer."
    )
