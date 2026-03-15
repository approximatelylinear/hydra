"""ColBERT head generator: direct FiLM adaptation in 128-dim token embedding space."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ColBERTHeadGenerator(nn.Module):
    """Given a task conditioning vector, generate FiLM modulation parameters
    for per-token ColBERT embeddings.

    Applies direct FiLM (no intermediate projection):
        modulated = gamma * token_emb + beta              # gamma/beta broadcast over seq dim
        adapted = token_emb + alpha * modulated            # alpha ≤ 0.3
        output = L2_normalize(adapted, dim=-1)             # per-token norm
    """

    def __init__(
        self,
        cond_dim: int = 256,
        embed_dim: int = 128,  # ColBERT token embedding dim
        rank: int = 32,  # unused, kept for API compatibility
    ):
        super().__init__()
        self.embed_dim = embed_dim

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
            Dict with keys 'gamma', 'beta', 'alpha'.
        """
        params = self.param_gen(cond)

        gamma = torch.sigmoid(params[:, : self._gamma_end]) * 2.0  # (batch, embed_dim)
        beta = params[:, self._gamma_end : self._beta_end]  # (batch, embed_dim)
        alpha = torch.sigmoid(params[:, self._beta_end :]) * 0.3  # (batch, 1), range [0, 0.3]

        return {"gamma": gamma, "beta": beta, "alpha": alpha}

    def apply_head(
        self, token_embeddings: torch.Tensor, head_params: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Apply FiLM modulation to per-token ColBERT embeddings.

        Args:
            token_embeddings: (batch, seq, embed_dim)
            head_params: output of forward()

        Returns:
            Adapted embeddings, L2-normalized per token. Shape (batch, seq, embed_dim).
        """
        batch_size = token_embeddings.size(0)
        gamma = head_params["gamma"]
        beta = head_params["beta"]
        alpha = head_params["alpha"]

        # Expand from (1, ...) to (batch, ...) if needed
        if gamma.size(0) == 1 and batch_size > 1:
            gamma = gamma.expand(batch_size, -1)
            beta = beta.expand(batch_size, -1)
            alpha = alpha.expand(batch_size, -1)

        # Direct FiLM: gamma and beta are (batch, embed_dim), broadcast over seq dim
        modulated = gamma.unsqueeze(1) * token_embeddings + beta.unsqueeze(1)

        # Mix: base + alpha * modulated, alpha is (batch, 1, 1)
        adapted = token_embeddings + alpha.unsqueeze(1) * modulated

        return F.normalize(adapted, p=2, dim=-1)
