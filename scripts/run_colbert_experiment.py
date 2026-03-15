#!/usr/bin/env python3
"""ColBERT multi-vector experiment: train hypernet with late interaction scoring.

- Frozen ColBERTv2 base encoder
- FiLM hypernet adaptation on per-token embeddings
- Teacher = BM25 + dense bi-encoder + reranker fused by RRF
- MaxSim late interaction scoring
- Evaluate: baseline ColBERT vs task-conditioned ColBERT
"""

import logging
import sys
from pathlib import Path

import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hydra.data.beir_loader import load_beir_dataset
from hydra.data.preference_pairs import generate_preference_pairs
from hydra.eval.colbert_evaluator import evaluate_colbert_retriever
from hydra.student.colbert_retriever import ColBERTRetriever
from hydra.teachers.bm25 import BM25Teacher
from hydra.teachers.dense import DenseTeacher
from hydra.teachers.ensemble import EnsembleTeacher
from hydra.teachers.jina_reranker import JinaRerankerTeacher
from hydra.training.colbert_trainer import ColBERTTrainConfig, train_colbert_hypernet

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# --- Config ---
TRAIN_DATASETS = [
    "scifact",
    "fiqa",
    "nfcorpus",
]

EVAL_DATASETS = [
    "scifact",
    "fiqa",
    "nfcorpus",
]

COLBERT_MODEL = "colbert-ir/colbertv2.0"
TASK_ENCODER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
COND_DIM = 256
EMBED_DIM = 128
CANDIDATES_PER_QUERY = 100
PAIRS_PER_QUERY = 16
HARD_NEGATIVE_RATIO = 0.5


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    # --- 1. Load training datasets ---
    logger.info("Loading training datasets...")
    train_datasets = {}
    for name in TRAIN_DATASETS:
        logger.info(f"  Loading {name}...")
        train_datasets[name] = load_beir_dataset(name)
        n_docs, n_queries = len(train_datasets[name].corpus), len(train_datasets[name].queries)
        logger.info(f"  {name}: {n_docs} docs, {n_queries} queries")

    # --- 2. Build teachers ---
    logger.info("Initializing teachers...")
    dense = DenseTeacher(model_name=TASK_ENCODER_MODEL)
    reranker = JinaRerankerTeacher()

    # --- 3. Generate preferences per task ---
    all_pairs = []

    for name, ds in train_datasets.items():
        logger.info(f"\n--- Generating teacher preferences for {name} ---")

        bm25 = BM25Teacher()
        bm25.index(ds.corpus_texts)

        dense.index(ds.corpus_texts)
        reranker.index(ds.corpus_texts)

        ensemble = EnsembleTeacher()
        ensemble.teachers = [bm25, dense, reranker]
        ensemble.weights = [1.0, 1.0, 3.0]

        query_texts = list(ds.queries.values())
        pairs = generate_preference_pairs(
            queries=query_texts,
            corpus_texts=ds.corpus_texts,
            teacher=ensemble,
            candidates_per_query=CANDIDATES_PER_QUERY,
            pairs_per_query=PAIRS_PER_QUERY,
            hard_negative_ratio=HARD_NEGATIVE_RATIO,
            task_name=name,
        )
        logger.info(f"  Generated {len(pairs)} preference pairs for {name}")
        all_pairs.extend(pairs)

    logger.info(f"\nTotal preference pairs across all tasks: {len(all_pairs)}")

    # --- 4. Train ColBERT hypernet ---
    logger.info("\n--- Training ColBERT hypernet ---")
    retriever = ColBERTRetriever(
        base_model=COLBERT_MODEL,
        task_encoder_model=TASK_ENCODER_MODEL,
        cond_dim=COND_DIM,
        embed_dim=EMBED_DIM,
    )

    task_cards = {}
    for name, ds in train_datasets.items():
        task_cards[name] = ds.task_card.to_text() if ds.task_card else name

    config = ColBERTTrainConfig(
        lr=1e-4,
        batch_size=16,
        epochs=10,
        device=device,
    )

    retriever = train_colbert_hypernet(
        retriever=retriever,
        pairs=all_pairs,
        task_cards=task_cards,
        config=config,
    )

    # --- 5. Load eval datasets ---
    eval_datasets = {}
    for name in EVAL_DATASETS:
        if name in train_datasets:
            eval_datasets[name] = train_datasets[name]
        else:
            logger.info(f"  Loading eval dataset {name}...")
            eval_datasets[name] = load_beir_dataset(name)

    # --- 6. Baseline: frozen ColBERT, no hypernet ---
    # Move trained retriever to CPU temporarily to free GPU for baseline eval
    retriever = retriever.cpu()
    torch.cuda.empty_cache()

    logger.info("\n--- Evaluation (baseline: frozen ColBERT, no FiLM) ---")
    logger.info(f"{'Dataset':15s} | {'MRR@10':>8s} | {'NDCG@10':>8s} | {'Recall@100':>10s}")
    logger.info("-" * 55)

    baseline = ColBERTRetriever(
        base_model=COLBERT_MODEL,
        task_encoder_model=TASK_ENCODER_MODEL,
        cond_dim=COND_DIM,
        embed_dim=EMBED_DIM,
    )
    baseline = baseline.to(device)
    for name, ds in eval_datasets.items():
        result = evaluate_colbert_retriever(
            baseline, ds, apply_head_to_docs=False,
        )
        logger.info(str(result))

    # Free baseline model before loading trained model back
    del baseline
    torch.cuda.empty_cache()

    # --- 7. Task-conditioned ColBERT ---
    retriever = retriever.to(device)
    logger.info("\n--- Evaluation (task-conditioned ColBERT) ---")
    logger.info(f"{'Dataset':15s} | {'MRR@10':>8s} | {'NDCG@10':>8s} | {'Recall@100':>10s}")
    logger.info("-" * 55)

    for name, ds in eval_datasets.items():
        result = evaluate_colbert_retriever(retriever, ds, apply_head_to_docs=True)
        logger.info(str(result))


if __name__ == "__main__":
    main()
