"""End-to-end evaluation for ColBERT multi-vector retriever."""

from __future__ import annotations

import logging

import torch

from hydra.data.beir_loader import RetrievalDataset
from hydra.eval.evaluator import EvalResult
from hydra.eval.metrics import mrr_at_k, ndcg_at_k, recall_at_k
from hydra.student.colbert_retriever import ColBERTRetriever, batched_maxsim

logger = logging.getLogger(__name__)


def evaluate_colbert_retriever(
    retriever: ColBERTRetriever,
    dataset: RetrievalDataset,
    task_text: str | None = None,
    apply_head_to_docs: bool = False,
    max_query_len: int = 32,
    max_doc_len: int = 180,
    batch_size: int = 64,
) -> EvalResult:
    """Evaluate ColBERT retriever on a BEIR-format dataset.

    Args:
        retriever: ColBERT retriever with trained hypernet.
        dataset: BEIR dataset to evaluate on.
        task_text: Task card text. If None, uses dataset's task card.
        apply_head_to_docs: If True, apply FiLM to docs too (slower but more accurate).
            At inference time, docs are nearly task-independent since alpha ≤ 0.3.
        max_query_len: Max query token length.
        max_doc_len: Max doc token length.
        batch_size: Encoding batch size.
    """
    retriever.eval()
    device = next(retriever.parameters()).device
    if task_text is None:
        task_text = dataset.task_card.to_text() if dataset.task_card else dataset.name

    with torch.no_grad():
        head_params = retriever.compile_task(task_text)

        # Encode corpus — store on CPU to avoid OOM on large corpora
        logger.info(f"Encoding {len(dataset.corpus_texts)} docs...")
        doc_chunks = []  # list of (embs_cpu, mask_cpu) tuples
        max_seq = 0
        for i in range(0, len(dataset.corpus_texts), batch_size):
            batch_texts = dataset.corpus_texts[i : i + batch_size]
            if apply_head_to_docs:
                embs, mask = retriever.encode_multi_vector(
                    batch_texts, head_params,
                    max_length=max_doc_len, is_query=False, batch_size=len(batch_texts),
                )
            else:
                embs, mask = retriever.base_encoder(
                    batch_texts, max_length=max_doc_len, is_query=False
                )
            max_seq = max(max_seq, embs.size(1))
            doc_chunks.append((embs.cpu(), mask.cpu()))

        # Encode queries (always apply FiLM) — small enough to stay on GPU
        query_ids = list(dataset.queries.keys())
        query_texts = [dataset.queries[qid] for qid in query_ids]
        logger.info(f"Encoding {len(query_texts)} queries...")
        q_embs, q_mask = retriever.encode_multi_vector(
            query_texts, head_params,
            max_length=max_query_len, is_query=True, batch_size=batch_size,
        )
        q_embs = q_embs.cpu()
        q_mask = q_mask.cpu()

    # Score all query-doc pairs via MaxSim
    # Process one query at a time against doc chunks to limit GPU memory
    logger.info("Computing MaxSim scores...")
    score_batch = 128  # docs per GPU scoring batch
    rankings = []

    for qi in range(len(query_ids)):
        q_single = q_embs[qi].unsqueeze(0)  # (1, q_len, dim)
        qm_single = q_mask[qi].unsqueeze(0)  # (1, q_len)

        all_scores = []
        for chunk_embs, chunk_mask in doc_chunks:
            # Pad chunk to max_seq if needed
            seq_len = chunk_embs.size(1)
            if seq_len < max_seq:
                pad_size = max_seq - seq_len
                chunk_embs = torch.nn.functional.pad(chunk_embs, (0, 0, 0, pad_size))
                chunk_mask = torch.nn.functional.pad(chunk_mask, (0, pad_size), value=False)

            # Score in sub-batches on GPU
            for di in range(0, chunk_embs.size(0), score_batch):
                d_batch = chunk_embs[di : di + score_batch].to(device)
                dm_batch = chunk_mask[di : di + score_batch].to(device)
                n_docs = d_batch.size(0)

                q_exp = q_single.expand(n_docs, -1, -1).to(device)
                qm_exp = qm_single.expand(n_docs, -1).to(device)

                scores = batched_maxsim(q_exp, d_batch, qm_exp, dm_batch)
                all_scores.append(scores.cpu())

        all_scores = torch.cat(all_scores)  # (n_docs,)
        top_indices = torch.argsort(-all_scores)[:100].numpy()
        rankings.append([dataset.corpus_ids[idx] for idx in top_indices])

        if (qi + 1) % 100 == 0:
            logger.info(f"  Scored {qi + 1}/{len(query_ids)} queries")

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
