"""Task card encoder: maps a task card to a fixed-size conditioning vector."""

from __future__ import annotations

import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer


class TaskCardEncoder(nn.Module):
    """Encode a task card text into a conditioning vector.

    Uses a frozen sentence-transformer to get an initial embedding,
    then projects to a conditioning dimension.
    """

    def __init__(
        self,
        base_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        cond_dim: int = 256,
    ):
        super().__init__()
        self.encoder = SentenceTransformer(base_model)
        # Freeze the base encoder — we only train the projection
        for p in self.encoder.parameters():
            p.requires_grad = False

        embed_dim = self.encoder.get_sentence_embedding_dimension()
        assert embed_dim is not None, f"Could not determine embedding dim for {base_model}"
        self.projection = nn.Sequential(
            nn.Linear(embed_dim, cond_dim),
            nn.GELU(),
            nn.Linear(cond_dim, cond_dim),
            nn.LayerNorm(cond_dim),
        )

    def forward(self, task_texts: list[str]) -> torch.Tensor:
        """Encode task card texts -> (batch, cond_dim) conditioning vectors."""
        with torch.no_grad():
            base_embs = self.encoder.encode(
                task_texts, convert_to_tensor=True, normalize_embeddings=True
            )
        return self.projection(base_embs)
