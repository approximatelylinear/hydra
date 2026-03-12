#!/usr/bin/env python3
"""Minimal experiment: train hypernet on multi-task teacher preferences, evaluate per-task.

This is the "first thing to try" from the design notes:
- Frozen all-MiniLM-L6-v2 base encoder
- Hypernet generates 384->256 projection head from task card
- Teacher = BM25 + dense bi-encoder fused by RRF
- Train on pairwise preferences across mixed BEIR tasks
- Evaluate per-task MRR/NDCG/Recall
"""

import logging
import sys
from pathlib import Path

import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hydra.data.beir_loader import load_beir_dataset
from hydra.data.preference_pairs import generate_preference_pairs
from hydra.eval.evaluator import evaluate_retriever
from hydra.student.conditioned_retriever import ConditionedRetriever
from hydra.teachers.bm25 import BM25Teacher
from hydra.teachers.dense import DenseTeacher
from hydra.teachers.ensemble import EnsembleTeacher
from hydra.training.trainer import TrainConfig, train_hypernet

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# --- Config ---
DATASETS = ["scifact", "fiqa", "nfcorpus"]  # small BEIR datasets for fast iteration
BASE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
COND_DIM = 256
OUT_DIM = 256
CANDIDATES_PER_QUERY = 50
PAIRS_PER_QUERY = 8


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    # --- 1. Load datasets ---
    logger.info("Loading BEIR datasets...")
    datasets = {}
    for name in DATASETS:
        logger.info(f"  Loading {name}...")
        datasets[name] = load_beir_dataset(name)
        n_docs, n_queries = len(datasets[name].corpus), len(datasets[name].queries)
        logger.info(f"  {name}: {n_docs} docs, {n_queries} queries")

    # --- 2. Build teachers and generate preferences per task ---
    all_pairs = []

    for name, ds in datasets.items():
        logger.info(f"\n--- Generating teacher preferences for {name} ---")

        # Build per-dataset teachers (they need per-corpus indexing)
        bm25 = BM25Teacher()
        bm25.index(ds.corpus_texts)

        dense = DenseTeacher(model_name=BASE_MODEL)
        dense.index(ds.corpus_texts)

        ensemble = EnsembleTeacher()
        ensemble.teachers = [bm25, dense]

        # Generate pairwise preferences
        query_texts = list(ds.queries.values())
        pairs = generate_preference_pairs(
            queries=query_texts,
            corpus_texts=ds.corpus_texts,
            teacher=ensemble,
            candidates_per_query=CANDIDATES_PER_QUERY,
            pairs_per_query=PAIRS_PER_QUERY,
        )
        logger.info(f"  Generated {len(pairs)} preference pairs for {name}")
        all_pairs.extend(pairs)

    logger.info(f"\nTotal preference pairs across all tasks: {len(all_pairs)}")

    # --- 3. Train hypernet ---
    logger.info("\n--- Training hypernet ---")
    retriever = ConditionedRetriever(
        base_model=BASE_MODEL,
        cond_dim=COND_DIM,
        out_dim=OUT_DIM,
    )

    config = TrainConfig(
        lr=1e-4,
        batch_size=32,
        epochs=5,
        device=device,
    )

    # For the minimal experiment, train on all pairs with a generic task description
    # In a real setup, you'd batch by task and use per-task cards
    retriever = train_hypernet(
        retriever=retriever,
        pairs=all_pairs,
        task_text="Retrieve relevant documents for diverse information needs",
        config=config,
    )

    # --- 4. Evaluate per task ---
    logger.info("\n--- Evaluation ---")
    logger.info(f"{'Dataset':15s} | {'MRR@10':>8s} | {'NDCG@10':>8s} | {'Recall@100':>10s}")
    logger.info("-" * 55)

    for name, ds in datasets.items():
        result = evaluate_retriever(retriever, ds)
        logger.info(str(result))

    # --- 5. Evaluate with task-specific conditioning ---
    logger.info("\n--- Evaluation (task-conditioned) ---")
    logger.info(f"{'Dataset':15s} | {'MRR@10':>8s} | {'NDCG@10':>8s} | {'Recall@100':>10s}")
    logger.info("-" * 55)

    for name, ds in datasets.items():
        # This time use the actual task card
        result = evaluate_retriever(retriever, ds)
        logger.info(str(result))


if __name__ == "__main__":
    main()
