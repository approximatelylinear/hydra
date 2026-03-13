"""Task card encoder: maps a task card to a fixed-size conditioning vector."""

from __future__ import annotations

import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer


class AttentionPooling(nn.Module):
    """Learned attention pooling over a variable number of embeddings."""

    def __init__(self, dim: int):
        super().__init__()
        self.attn = nn.Linear(dim, 1)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Pool (n, dim) -> (dim,) via learned attention weights."""
        weights = torch.softmax(self.attn(embeddings), dim=0)  # (n, 1)
        return (weights * embeddings).sum(dim=0)  # (dim,)


class TaskCardEncoder(nn.Module):
    """Encode a task card text into a conditioning vector.

    Instead of encoding the entire task card as one string, splits it into
    individual lines (description, exemplars, etc.) and encodes each separately.
    An attention pooling layer learns to weight the most informative lines,
    giving the hypernet distributional signal from exemplars rather than just
    a single-string embedding.
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

        self.attn_pool = AttentionPooling(embed_dim)
        self.projection = nn.Sequential(
            nn.Linear(embed_dim, cond_dim),
            nn.GELU(),
            nn.Linear(cond_dim, cond_dim),
            nn.LayerNorm(cond_dim),
        )

    def _split_task_text(self, text: str) -> list[str]:
        """Split task card text into meaningful lines for separate encoding."""
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        # Filter out section headers that aren't informative on their own
        lines = [l for l in lines if l not in ("Example queries:", "Example documents:")]
        return lines if lines else [text]

    def forward(self, task_texts: list[str]) -> torch.Tensor:
        """Encode task card texts -> (batch, cond_dim) conditioning vectors."""
        batch_results = []
        for text in task_texts:
            lines = self._split_task_text(text)
            with torch.no_grad():
                line_embs = self.encoder.encode(
                    lines, convert_to_tensor=True, normalize_embeddings=True
                ).clone()
            pooled = self.attn_pool(line_embs)  # (embed_dim,)
            batch_results.append(pooled)

        stacked = torch.stack(batch_results)  # (batch, embed_dim)
        return self.projection(stacked)
