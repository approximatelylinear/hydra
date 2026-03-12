"""Projection head generator: hypernet that produces per-task residual adaptations."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHeadGenerator(nn.Module):
    """Given a task conditioning vector, generate a residual adaptation head.

    Instead of replacing the base embedding with a lossy projection, this generates
    a small residual delta that *adjusts* the base embedding. The final output is:

        output = normalize(base_embedding + alpha * residual)

    where both the residual transform and alpha are generated from the task card.
    With an untrained hypernet, alpha starts near zero -> baseline performance.
    Training can only improve from there.
    """

    def __init__(
        self,
        cond_dim: int = 256,
        embed_dim: int = 384,  # all-MiniLM-L6-v2 output dim
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # Low-rank residual: generate two smaller matrices A (embed_dim x rank)
        # and B (rank x embed_dim) so the residual is x @ A @ B + bias
        # This keeps param count manageable vs full embed_dim x embed_dim
        rank = 64
        self.rank = rank

        n_A_params = embed_dim * rank
        n_B_params = rank * embed_dim
        n_bias_params = embed_dim
        n_alpha = 1  # scalar mixing weight

        total_params = n_A_params + n_B_params + n_bias_params + n_alpha

        self.param_gen = nn.Sequential(
            nn.Linear(cond_dim, cond_dim * 2),
            nn.GELU(),
            nn.Linear(cond_dim * 2, cond_dim * 2),
            nn.GELU(),
            nn.Linear(cond_dim * 2, total_params),
        )

        # Initialize final layer to near-zero so alpha starts small
        final_layer = self.param_gen[-1]
        assert isinstance(final_layer, nn.Linear)
        nn.init.zeros_(final_layer.weight)
        nn.init.zeros_(final_layer.bias)

        self._A_end = n_A_params
        self._B_end = n_A_params + n_B_params
        self._bias_end = self._B_end + n_bias_params

    def forward(self, cond: torch.Tensor) -> dict[str, torch.Tensor]:
        """Generate residual head parameters from conditioning vector.

        Args:
            cond: (batch, cond_dim) conditioning vectors from TaskCardEncoder

        Returns:
            Dict with keys 'A', 'B', 'bias', 'alpha' — each batched.
        """
        params = self.param_gen(cond)

        A = params[:, : self._A_end].view(-1, self.embed_dim, self.rank)
        B = params[:, self._A_end : self._B_end].view(-1, self.rank, self.embed_dim)
        bias = params[:, self._B_end : self._bias_end]
        alpha = torch.sigmoid(params[:, self._bias_end :])  # (batch, 1), bounded [0, 1]

        return {"A": A, "B": B, "bias": bias, "alpha": alpha}

    @staticmethod
    def apply_head(embeddings: torch.Tensor, head_params: dict[str, torch.Tensor]) -> torch.Tensor:
        """Apply residual adaptation to base embeddings.

        Args:
            embeddings: (batch, embed_dim)
            head_params: output of forward()

        Returns:
            Adapted embeddings, L2-normalized. Same dimensionality as input.
        """
        batch_size = embeddings.size(0)
        A = head_params["A"]
        B = head_params["B"]
        bias = head_params["bias"]
        alpha = head_params["alpha"]

        # Expand from (1, ...) to (batch, ...) if needed
        if A.size(0) == 1 and batch_size > 1:
            A = A.expand(batch_size, -1, -1)
            B = B.expand(batch_size, -1, -1)
            bias = bias.expand(batch_size, -1)
            alpha = alpha.expand(batch_size, -1)

        # Low-rank residual: x @ A @ B + bias
        # (batch, 1, embed_dim) @ (batch, embed_dim, rank) -> (batch, 1, rank)
        # (batch, 1, rank) @ (batch, rank, embed_dim) -> (batch, 1, embed_dim)
        residual = torch.bmm(torch.bmm(embeddings.unsqueeze(1), A), B).squeeze(1) + bias

        # Mix: base + alpha * residual
        adapted = embeddings + alpha * residual

        return F.normalize(adapted, p=2, dim=-1)
