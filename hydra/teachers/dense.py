"""Dense bi-encoder teacher using a frozen sentence-transformer."""

from __future__ import annotations

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


class DenseTeacher:
    """Frozen bi-encoder that scores query-doc pairs by cosine similarity."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.model.eval()
        self.doc_embeddings: np.ndarray | None = None

    @torch.no_grad()
    def index(self, corpus: list[str], batch_size: int = 256) -> None:
        self.doc_embeddings = self.model.encode(
            corpus, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True
        )

    @torch.no_grad()
    def rank(self, query: str, doc_indices: list[int] | None = None) -> np.ndarray:
        """Return cosine similarity scores for docs (or a subset)."""
        assert self.doc_embeddings is not None, "Call .index(corpus) first"
        q_emb = self.model.encode([query], normalize_embeddings=True)
        embs = self.doc_embeddings
        if doc_indices is not None:
            embs = embs[doc_indices]
        return (q_emb @ embs.T).flatten()
