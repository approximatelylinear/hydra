"""Ensemble teacher: fuse multiple teachers via Reciprocal Rank Fusion."""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field

import numpy as np


@dataclass
class EnsembleTeacher:
    """Fuses teacher rankings using RRF.

    Each teacher must implement .rank(query, doc_indices) -> scores array.
    """

    teachers: list = field(default_factory=list)
    k: int = 60  # RRF constant

    def add_teacher(self, teacher) -> None:
        self.teachers.append(teacher)

    def index(self, corpus: list[str], **kwargs) -> None:
        for t in self.teachers:
            sig = inspect.signature(t.index)
            supported = {k: v for k, v in kwargs.items() if k in sig.parameters}
            t.index(corpus, **supported)

    def _rrf_scores(self, rankings: list[np.ndarray], n_docs: int) -> np.ndarray:
        """Compute RRF scores from multiple score arrays."""
        fused = np.zeros(n_docs)
        for scores in rankings:
            order = np.argsort(-scores)
            for rank, idx in enumerate(order):
                fused[idx] += 1.0 / (self.k + rank + 1)
        return fused

    def rank(self, query: str, doc_indices: list[int] | None = None) -> np.ndarray:
        """Return RRF-fused scores."""
        rankings = [t.rank(query, doc_indices) for t in self.teachers]
        n_docs = len(rankings[0])
        return self._rrf_scores(rankings, n_docs)

    def rank_pairwise(
        self, query: str, doc_indices: list[int] | None = None
    ) -> list[tuple[int, int]]:
        """Return ordered pairs (winner, loser) from fused ranking."""
        scores = self.rank(query, doc_indices)
        order = np.argsort(-scores)
        pairs = []
        for i in range(len(order)):
            for j in range(i + 1, min(i + 10, len(order))):  # top-k window
                pairs.append((int(order[i]), int(order[j])))
        return pairs
