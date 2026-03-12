"""Generate pairwise preference data from teacher rankings."""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np


@dataclass
class PreferencePair:
    """A single pairwise preference: doc_pos is preferred over doc_neg for query."""

    query: str
    doc_pos: str
    doc_neg: str
    margin: float = 0.0  # teacher score difference (for weighting)


def generate_preference_pairs(
    queries: list[str],
    corpus_texts: list[str],
    teacher,
    candidates_per_query: int = 100,
    pairs_per_query: int = 10,
    seed: int = 42,
) -> list[PreferencePair]:
    """Generate pairwise preferences from teacher rankings.

    For each query:
    1. Sample candidate docs (or use all if corpus is small)
    2. Get teacher scores
    3. Extract pairs with margin filtering

    This distills orderings, not raw scores — robust to teacher miscalibration.
    """
    rng = random.Random(seed)
    n_docs = len(corpus_texts)
    pairs = []

    for query in queries:
        # Sample candidate indices
        if n_docs <= candidates_per_query:
            doc_indices = list(range(n_docs))
        else:
            doc_indices = rng.sample(range(n_docs), candidates_per_query)

        # Get teacher scores for candidates
        scores = teacher.rank(query, doc_indices)
        ranked = np.argsort(-scores)

        # Generate pairs: top vs bottom half with margin
        n = len(ranked)
        top_half = ranked[: n // 4]  # top quartile
        bottom_half = ranked[n // 2 :]  # bottom half

        query_pairs = []
        for _ in range(pairs_per_query):
            pos_rank_idx = rng.choice(top_half)
            neg_rank_idx = rng.choice(bottom_half)

            pos_doc_idx = doc_indices[pos_rank_idx]
            neg_doc_idx = doc_indices[neg_rank_idx]

            margin = float(scores[pos_rank_idx] - scores[neg_rank_idx])
            if margin > 0:
                query_pairs.append(
                    PreferencePair(
                        query=query,
                        doc_pos=corpus_texts[pos_doc_idx],
                        doc_neg=corpus_texts[neg_doc_idx],
                        margin=margin,
                    )
                )

        pairs.extend(query_pairs)

    return pairs
