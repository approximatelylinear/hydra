"""Projection head generator: FiLM-conditioned residual adaptation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHeadGenerator(nn.Module):
    """Given a task conditioning vector, generate FiLM modulation parameters.

    Instead of generating full weight matrices per task, this uses shared learned
    low-rank residual matrices (A, B) and generates only lightweight FiLM
    scale/shift parameters (gamma, beta) plus a mixing weight (alpha) per task.

    The final output is:
        residual = x @ A_shared @ B_shared
        modulated = gamma * residual + beta
        output = normalize(x + alpha * modulated)

    This is much easier to train with few tasks — the hypernet only needs to learn
    how to modulate, not how to construct an entire linear transform.
    """

    def __init__(
        self,
        cond_dim: int = 256,
        embed_dim: int = 384,  # all-MiniLM-L6-v2 output dim
        rank: int = 128,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.rank = rank

        # Frozen orthogonal projection (fixed feature extractor, not learned)
        # FiLM modulation learns to select/combine these fixed features per task.
        # Freezing prevents the collapse seen when A/B drift during extended training.
        A_init = torch.nn.init.orthogonal_(torch.empty(embed_dim, rank))
        B_init = torch.nn.init.orthogonal_(torch.empty(rank, embed_dim))
        self.register_buffer("A_shared", A_init)
        self.register_buffer("B_shared", B_init)

        # FiLM parameter generator: produce gamma, beta, alpha from conditioning
        n_film_params = embed_dim + embed_dim + 1  # gamma + beta + alpha
        self.param_gen = nn.Sequential(
            nn.Linear(cond_dim, cond_dim * 2),
            nn.GELU(),
            nn.Linear(cond_dim * 2, cond_dim * 2),
            nn.GELU(),
            nn.Linear(cond_dim * 2, n_film_params),
        )

        # Initialize final layer so gamma≈1, beta≈0, alpha≈0
        final_layer = self.param_gen[-1]
        assert isinstance(final_layer, nn.Linear)
        nn.init.zeros_(final_layer.weight)
        # gamma bias = 0 → sigmoid(0) * 2 = 1.0 (identity scale)
        # beta bias = 0 (no shift)
        # alpha bias = -2 → sigmoid(-2) ≈ 0.12 (small initial mixing)
        bias_init = torch.zeros(n_film_params)
        bias_init[-1] = -2.0  # alpha starts small
        final_layer.bias = nn.Parameter(bias_init)

        self._gamma_end = embed_dim
        self._beta_end = embed_dim * 2

    def forward(self, cond: torch.Tensor) -> dict[str, torch.Tensor]:
        """Generate FiLM modulation parameters from conditioning vector.

        Args:
            cond: (batch, cond_dim) conditioning vectors from TaskCardEncoder

        Returns:
            Dict with keys 'gamma', 'beta', 'alpha', 'A', 'B'.
        """
        params = self.param_gen(cond)

        # gamma: scale factor, centered around 1 via sigmoid * 2
        gamma = torch.sigmoid(params[:, : self._gamma_end]) * 2.0  # (batch, embed_dim), range [0, 2]
        beta = params[:, self._gamma_end : self._beta_end]  # (batch, embed_dim)
        alpha = torch.sigmoid(params[:, self._beta_end :]) * 0.3  # (batch, 1), range [0, 0.3]

        # Expand shared matrices to batch dim
        batch_size = cond.size(0)
        A = self.A_shared.unsqueeze(0).expand(batch_size, -1, -1)
        B = self.B_shared.unsqueeze(0).expand(batch_size, -1, -1)

        return {"A": A, "B": B, "gamma": gamma, "beta": beta, "alpha": alpha}

    @staticmethod
    def apply_head(embeddings: torch.Tensor, head_params: dict[str, torch.Tensor]) -> torch.Tensor:
        """Apply FiLM-modulated residual adaptation to base embeddings.

        Args:
            embeddings: (batch, embed_dim)
            head_params: output of forward()

        Returns:
            Adapted embeddings, L2-normalized. Same dimensionality as input.
        """
        batch_size = embeddings.size(0)
        A = head_params["A"]
        B = head_params["B"]
        gamma = head_params["gamma"]
        beta = head_params["beta"]
        alpha = head_params["alpha"]

        # Expand from (1, ...) to (batch, ...) if needed
        if A.size(0) == 1 and batch_size > 1:
            A = A.expand(batch_size, -1, -1)
            B = B.expand(batch_size, -1, -1)
            gamma = gamma.expand(batch_size, -1)
            beta = beta.expand(batch_size, -1)
            alpha = alpha.expand(batch_size, -1)

        # Shared low-rank residual: x @ A @ B
        residual = torch.bmm(torch.bmm(embeddings.unsqueeze(1), A), B).squeeze(1)

        # FiLM modulation: gamma * residual + beta
        modulated = gamma * residual + beta

        # Mix: base + alpha * modulated
        adapted = embeddings + alpha * modulated

        return F.normalize(adapted, p=2, dim=-1)
