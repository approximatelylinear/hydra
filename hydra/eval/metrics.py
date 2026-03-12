"""Retrieval evaluation metrics."""

from __future__ import annotations

import numpy as np


def mrr_at_k(rankings: list[list[str]], qrels: dict[str, dict[str, int]], k: int = 10) -> float:
    """Mean Reciprocal Rank @ k.

    Args:
        rankings: list of ranked doc_id lists (one per query)
        qrels: query_id -> {doc_id: relevance} ground truth
        k: cutoff
    """
    query_ids = list(qrels.keys())
    rrs = []
    for qid, ranking in zip(query_ids, rankings):
        relevant = set(did for did, rel in qrels[qid].items() if rel > 0)
        for rank, did in enumerate(ranking[:k], 1):
            if did in relevant:
                rrs.append(1.0 / rank)
                break
        else:
            rrs.append(0.0)
    return float(np.mean(rrs))


def recall_at_k(rankings: list[list[str]], qrels: dict[str, dict[str, int]], k: int = 100) -> float:
    """Recall @ k."""
    query_ids = list(qrels.keys())
    recalls = []
    for qid, ranking in zip(query_ids, rankings):
        relevant = set(did for did, rel in qrels[qid].items() if rel > 0)
        if not relevant:
            continue
        retrieved = set(ranking[:k])
        recalls.append(len(relevant & retrieved) / len(relevant))
    return float(np.mean(recalls))


def ndcg_at_k(rankings: list[list[str]], qrels: dict[str, dict[str, int]], k: int = 10) -> float:
    """NDCG @ k."""
    query_ids = list(qrels.keys())
    ndcgs = []

    for qid, ranking in zip(query_ids, rankings):
        rels = qrels.get(qid, {})
        dcg = 0.0
        for rank, did in enumerate(ranking[:k], 1):
            rel = rels.get(did, 0)
            dcg += (2**rel - 1) / np.log2(rank + 1)

        # Ideal DCG
        ideal_rels = sorted(rels.values(), reverse=True)[:k]
        idcg = sum((2**r - 1) / np.log2(rank + 1) for rank, r in enumerate(ideal_rels, 1))

        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)

    return float(np.mean(ndcgs))
