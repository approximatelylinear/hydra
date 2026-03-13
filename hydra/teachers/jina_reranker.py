"""Jina Reranker v3 teacher: 0.6B parameter reranker with listwise attention."""

from __future__ import annotations

import numpy as np
import torch


class JinaRerankerTeacher:
    """Jina Reranker v3 — scores query-doc pairs using causal listwise attention.

    Built on Qwen3-0.6B with a "Last but Not Late Interaction" architecture.
    Can process up to 64 documents simultaneously within one context window,
    making it significantly faster than pairwise cross-encoders for batched scoring.
    """

    def __init__(
        self,
        model_name: str = "jinaai/jina-reranker-v3",
        device: str | None = None,
    ):
        from transformers import AutoModel

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            trust_remote_code=True,
            tie_word_embeddings=False,
        ).to(device)
        self.model.eval()
        self.device = device
        self.corpus: list[str] | None = None

    def index(self, corpus: list[str]) -> None:
        """Store corpus texts."""
        self.corpus = corpus

    @torch.no_grad()
    def rank(
        self,
        query: str,
        doc_indices: list[int] | None = None,
    ) -> np.ndarray:
        """Score query against docs using Jina reranker.

        Args:
            query: Query text.
            doc_indices: If provided, only score these corpus indices.

        Returns:
            Array of relevance scores, in the same order as doc_indices
            (or full corpus order if doc_indices is None).
        """
        assert self.corpus is not None, "Call .index(corpus) first"

        if doc_indices is not None:
            docs = [self.corpus[i] for i in doc_indices]
        else:
            docs = list(self.corpus)

        results = self.model.rerank(query, docs)

        # Results come sorted by score — rebuild array in original doc order
        scores = np.zeros(len(docs), dtype=np.float32)
        for r in results:
            scores[r["index"]] = r["relevance_score"]

        return scores
