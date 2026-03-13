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
    task_name: str = ""  # which task this pair belongs to


def generate_preference_pairs(
    queries: list[str],
    corpus_texts: list[str],
    teacher,
    candidates_per_query: int = 100,
    pairs_per_query: int = 10,
    hard_negative_ratio: float = 0.5,
    task_name: str = "",
    seed: int = 42,
) -> list[PreferencePair]:
    """Generate pairwise preferences from teacher rankings.

    For each query:
    1. Sample candidate docs (or use all if corpus is small)
    2. Get teacher scores
    3. Extract pairs — mix of hard negatives (near the decision boundary)
       and easy negatives (for stable gradients)

    Args:
        hard_negative_ratio: Fraction of pairs that use hard negatives
            (rank 5-20 region). Rest use easy negatives (bottom half).
    """
    rng = random.Random(seed)
    n_docs = len(corpus_texts)
    pairs = []

    n_hard = int(pairs_per_query * hard_negative_ratio)
    n_easy = pairs_per_query - n_hard

    for query in queries:
        # Sample candidate indices
        if n_docs <= candidates_per_query:
            doc_indices = list(range(n_docs))
        else:
            doc_indices = rng.sample(range(n_docs), candidates_per_query)

        # Get teacher scores for candidates
        scores = teacher.rank(query, doc_indices)
        ranked = np.argsort(-scores)

        n = len(ranked)
        top_docs = ranked[:3]  # top-3 positives
        hard_neg_pool = ranked[5:20] if n > 20 else ranked[3 : n // 2]  # near misses
        easy_neg_pool = ranked[n // 2 :]  # bottom half

        if len(hard_neg_pool) == 0 or len(easy_neg_pool) == 0 or len(top_docs) == 0:
            continue

        query_pairs = []

        # Hard negatives: top-3 vs rank 5-20
        for _ in range(n_hard):
            pos_rank_idx = rng.choice(top_docs)
            neg_rank_idx = rng.choice(hard_neg_pool)
            _maybe_add_pair(
                query,
                scores,
                pos_rank_idx,
                neg_rank_idx,
                doc_indices,
                corpus_texts,
                task_name,
                query_pairs,
            )

        # Easy negatives: top-3 vs bottom half
        for _ in range(n_easy):
            pos_rank_idx = rng.choice(top_docs)
            neg_rank_idx = rng.choice(easy_neg_pool)
            _maybe_add_pair(
                query,
                scores,
                pos_rank_idx,
                neg_rank_idx,
                doc_indices,
                corpus_texts,
                task_name,
                query_pairs,
            )

        pairs.extend(query_pairs)

    return pairs


def _maybe_add_pair(
    query: str,
    scores: np.ndarray,
    pos_rank_idx: int,
    neg_rank_idx: int,
    doc_indices: list[int],
    corpus_texts: list[str],
    task_name: str,
    out: list[PreferencePair],
) -> None:
    """Add a pair if the margin is positive."""
    margin = float(scores[pos_rank_idx] - scores[neg_rank_idx])
    if margin > 0:
        out.append(
            PreferencePair(
                query=query,
                doc_pos=corpus_texts[doc_indices[pos_rank_idx]],
                doc_neg=corpus_texts[doc_indices[neg_rank_idx]],
                margin=margin,
                task_name=task_name,
            )
        )
