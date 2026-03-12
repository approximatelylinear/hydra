"""End-to-end evaluation: encode corpus, retrieve, compute metrics."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch

from hydra.data.beir_loader import RetrievalDataset
from hydra.eval.metrics import mrr_at_k, ndcg_at_k, recall_at_k
from hydra.student.conditioned_retriever import ConditionedRetriever

logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    dataset: str
    mrr_10: float
    ndcg_10: float
    recall_100: float

    def __str__(self):
        return (
            f"{self.dataset:15s} | MRR@10: {self.mrr_10:.4f} | "
            f"NDCG@10: {self.ndcg_10:.4f} | Recall@100: {self.recall_100:.4f}"
        )


def evaluate_retriever(
    retriever: ConditionedRetriever,
    dataset: RetrievalDataset,
    batch_size: int = 256,
) -> EvalResult:
    """Evaluate retriever on a BEIR-format dataset."""
    retriever.eval()
    task_text = dataset.task_card.to_text() if dataset.task_card else dataset.name

    with torch.no_grad():
        head_params = retriever.compile_task(task_text)

        # Encode corpus
        logger.info(f"Encoding {len(dataset.corpus_texts)} docs...")
        doc_embs = retriever.encode(dataset.corpus_texts, head_params, batch_size=batch_size)

        # Encode queries
        query_ids = list(dataset.queries.keys())
        query_texts = [dataset.queries[qid] for qid in query_ids]
        logger.info(f"Encoding {len(query_texts)} queries...")
        q_embs = retriever.encode(query_texts, head_params, batch_size=batch_size)

    # Retrieve top-100 by cosine similarity
    doc_embs_np = doc_embs.cpu().numpy()
    q_embs_np = q_embs.cpu().numpy()
    sim_matrix = q_embs_np @ doc_embs_np.T

    rankings = []
    for i in range(len(query_ids)):
        top_indices = np.argsort(-sim_matrix[i])[:100]
        rankings.append([dataset.corpus_ids[idx] for idx in top_indices])

    # Filter to queries that have qrels
    filtered_rankings = []
    filtered_qrels = {}
    for qid, ranking in zip(query_ids, rankings):
        if qid in dataset.qrels:
            filtered_rankings.append(ranking)
            filtered_qrels[qid] = dataset.qrels[qid]

    return EvalResult(
        dataset=dataset.name,
        mrr_10=mrr_at_k(filtered_rankings, filtered_qrels, k=10),
        ndcg_10=ndcg_at_k(filtered_rankings, filtered_qrels, k=10),
        recall_100=recall_at_k(filtered_rankings, filtered_qrels, k=100),
    )
