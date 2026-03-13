"""Cross-encoder teacher: joint query-doc attention for precise relevance scoring."""

from __future__ import annotations

import numpy as np
import torch
from sentence_transformers import CrossEncoder


class CrossEncoderTeacher:
    """Cross-encoder that scores query-doc pairs with joint attention.

    Much stronger than bi-encoders for ranking since it models token-level
    query-doc interactions. Too slow for retrieval, but ideal as a teacher
    for offline preference generation.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)
        self.corpus: list[str] | None = None

    def index(self, corpus: list[str]) -> None:
        """Store corpus texts. No embedding precomputation — cross-encoders
        score each (query, doc) pair jointly."""
        self.corpus = corpus

    @torch.no_grad()
    def rank(
        self,
        query: str,
        doc_indices: list[int] | None = None,
        batch_size: int = 64,
    ) -> np.ndarray:
        """Score query against docs (or a subset) using cross-encoder.

        Args:
            query: Query text.
            doc_indices: If provided, only score these corpus indices.
            batch_size: Batch size for cross-encoder inference.

        Returns:
            Array of relevance scores.
        """
        assert self.corpus is not None, "Call .index(corpus) first"

        if doc_indices is not None:
            docs = [self.corpus[i] for i in doc_indices]
        else:
            docs = self.corpus

        pairs = [[query, doc] for doc in docs]
        scores = self.model.predict(pairs, batch_size=batch_size, show_progress_bar=False)
        return np.array(scores, dtype=np.float32)
