"""Projection head generator: hypernet that produces per-task embedding heads."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class ProjectionHeadGenerator(nn.Module):
    """Given a task conditioning vector, generate a lightweight projection head.

    The generated head maps base embeddings (embed_dim) to task-specific
    embeddings (out_dim) via a linear projection + learned normalization params.

    This is the core "hypernet" — it outputs weights, not activations.
    """

    def __init__(
        self,
        cond_dim: int = 256,
        embed_dim: int = 384,  # all-MiniLM-L6-v2 output dim
        out_dim: int = 256,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.out_dim = out_dim

        # Generate the projection matrix W (embed_dim x out_dim) and bias
        n_weight_params = embed_dim * out_dim
        n_bias_params = out_dim
        # Plus LayerNorm scale + shift
        n_norm_params = out_dim * 2

        total_params = n_weight_params + n_bias_params + n_norm_params

        self.param_gen = nn.Sequential(
            nn.Linear(cond_dim, cond_dim * 2),
            nn.GELU(),
            nn.Linear(cond_dim * 2, cond_dim * 2),
            nn.GELU(),
            nn.Linear(cond_dim * 2, total_params),
        )

        self._weight_end = n_weight_params
        self._bias_end = n_weight_params + n_bias_params
        self._scale_end = self._bias_end + out_dim

    def forward(self, cond: torch.Tensor) -> dict[str, torch.Tensor]:
        """Generate projection head parameters from conditioning vector.

        Args:
            cond: (batch, cond_dim) conditioning vectors from TaskCardEncoder

        Returns:
            Dict with keys 'weight', 'bias', 'scale', 'shift' — each batched.
        """
        params = self.param_gen(cond)

        W = rearrange(
            params[:, : self._weight_end],
            "b (d_in d_out) -> b d_in d_out",
            d_in=self.embed_dim,
            d_out=self.out_dim,
        )
        bias = params[:, self._weight_end : self._bias_end]
        scale = params[:, self._bias_end : self._scale_end]
        shift = params[:, self._scale_end :]

        return {"weight": W, "bias": bias, "scale": scale, "shift": shift}

    @staticmethod
    def apply_head(embeddings: torch.Tensor, head_params: dict[str, torch.Tensor]) -> torch.Tensor:
        """Apply a generated projection head to base embeddings.

        Args:
            embeddings: (batch, seq, embed_dim) or (batch, embed_dim)
            head_params: output of forward() — each value is (1, ...) or (batch, ...)

        Returns:
            Projected embeddings, L2-normalized.
        """
        batch_size = embeddings.size(0)
        W = head_params["weight"]  # (1 or batch, embed_dim, out_dim)
        bias = head_params["bias"]  # (1 or batch, out_dim)
        scale = head_params["scale"]
        shift = head_params["shift"]

        # Expand head params from (1, ...) to (batch, ...) if needed
        if W.size(0) == 1 and batch_size > 1:
            W = W.expand(batch_size, -1, -1)
            bias = bias.expand(batch_size, -1)
            scale = scale.expand(batch_size, -1)
            shift = shift.expand(batch_size, -1)

        if embeddings.dim() == 2:
            # (batch, embed_dim) @ (batch, embed_dim, out_dim) -> (batch, out_dim)
            projected = torch.bmm(embeddings.unsqueeze(1), W).squeeze(1) + bias
        else:
            # (batch, seq, embed_dim) @ (batch, embed_dim, out_dim) -> (batch, seq, out_dim)
            projected = torch.bmm(embeddings, W) + bias.unsqueeze(1)

        # Learned LayerNorm
        projected = F.layer_norm(projected, [projected.size(-1)])
        projected = projected * (1 + scale.unsqueeze(-2) if projected.dim() == 3 else 1 + scale)
        projected = projected + (shift.unsqueeze(-2) if projected.dim() == 3 else shift)

        return F.normalize(projected, p=2, dim=-1)
