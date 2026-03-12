"""BM25 teacher: lexical baseline for ensemble."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from rank_bm25 import BM25Okapi


@dataclass
class BM25Teacher:
    """Wraps BM25Okapi to produce rankings over a corpus."""

    corpus_tokens: list[list[str]] | None = None
    _index: BM25Okapi | None = None

    def index(self, corpus: list[str]) -> None:
        self.corpus_tokens = [doc.lower().split() for doc in corpus]
        self._index = BM25Okapi(self.corpus_tokens)

    def rank(self, query: str, doc_indices: list[int] | None = None) -> np.ndarray:
        """Return BM25 scores for all docs (or a subset by index)."""
        assert self._index is not None, "Call .index(corpus) first"
        scores = self._index.get_scores(query.lower().split())
        if doc_indices is not None:
            scores = scores[doc_indices]
        return scores
