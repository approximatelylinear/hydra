#!/usr/bin/env python3
"""Multi-task experiment: train hypernet on diverse BEIR tasks with hard negatives.

- Frozen all-MiniLM-L6-v2 base encoder
- Residual hypernet adaptation from task cards
- Teacher = BM25 + dense bi-encoder fused by RRF
- Hard negatives (50% near-miss, 50% easy) for sharper discrimination
- Per-task batching with task card conditioning
- Evaluate: baseline vs generic vs task-conditioned
"""

import logging
import sys
from pathlib import Path

import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hydra.data.beir_loader import load_beir_dataset
from hydra.data.preference_pairs import generate_preference_pairs
from hydra.eval.evaluator import evaluate_baseline, evaluate_retriever
from hydra.student.conditioned_retriever import ConditionedRetriever
from hydra.teachers.bm25 import BM25Teacher
from hydra.teachers.dense import DenseTeacher
from hydra.teachers.ensemble import EnsembleTeacher
from hydra.teachers.jina_reranker import JinaRerankerTeacher
from hydra.training.trainer import TrainConfig, train_hypernet

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# --- Config ---
# Training tasks: diverse mix of domains and query types
TRAIN_DATASETS = [
    "scifact",  # scientific claim verification
    "fiqa",  # financial QA
    "nfcorpus",  # health/nutrition
    "arguana",  # counter-argument retrieval
    "scidocs",  # scientific paper similarity
]

# Eval tasks: includes training tasks
EVAL_DATASETS = [
    "scifact",
    "fiqa",
    "nfcorpus",
    "arguana",
    "scidocs",
]

BASE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
COND_DIM = 256
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

    # --- 2. Build teachers (shared across datasets) ---
    logger.info("Initializing teachers...")
    dense = DenseTeacher(model_name=BASE_MODEL)
    reranker = JinaRerankerTeacher()

    # --- 3. Generate preferences per task ---
    all_pairs = []

    for name, ds in train_datasets.items():
        logger.info(f"\n--- Generating teacher preferences for {name} ---")

        bm25 = BM25Teacher()  # lightweight, per-corpus index is cheap
        bm25.index(ds.corpus_texts)

        dense.index(ds.corpus_texts)
        reranker.index(ds.corpus_texts)

        ensemble = EnsembleTeacher()
        ensemble.teachers = [bm25, dense, reranker]
        ensemble.weights = [1.0, 1.0, 3.0]  # reranker weighted 3x

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

    # --- 4. Train hypernet ---
    logger.info("\n--- Training hypernet (per-task batching, hard negatives) ---")
    retriever = ConditionedRetriever(
        base_model=BASE_MODEL,
        cond_dim=COND_DIM,
    )

    task_cards = {}
    for name, ds in train_datasets.items():
        task_cards[name] = ds.task_card.to_text() if ds.task_card else name

    config = TrainConfig(
        lr=1e-4,
        batch_size=32,
        epochs=15,
        device=device,
    )

    retriever = train_hypernet(
        retriever=retriever,
        pairs=all_pairs,
        task_cards=task_cards,
        config=config,
    )

    # --- 5. Load eval datasets (may overlap with training) ---
    eval_datasets = {}
    for name in EVAL_DATASETS:
        if name in train_datasets:
            eval_datasets[name] = train_datasets[name]
        else:
            logger.info(f"  Loading eval dataset {name}...")
            eval_datasets[name] = load_beir_dataset(name)

    # --- 6. Baseline: frozen encoder, no hypernet ---
    logger.info("\n--- Evaluation (baseline: frozen encoder, no hypernet) ---")
    logger.info(f"{'Dataset':15s} | {'MRR@10':>8s} | {'NDCG@10':>8s} | {'Recall@100':>10s}")
    logger.info("-" * 55)

    for name, ds in eval_datasets.items():
        result = evaluate_baseline(BASE_MODEL, ds)
        logger.info(str(result))

    # --- 7. Generic conditioning ---
    generic_task = "Retrieve relevant documents for diverse information needs"
    logger.info("\n--- Evaluation (generic conditioning) ---")
    logger.info(f"{'Dataset':15s} | {'MRR@10':>8s} | {'NDCG@10':>8s} | {'Recall@100':>10s}")
    logger.info("-" * 55)

    for name, ds in eval_datasets.items():
        result = evaluate_retriever(retriever, ds, task_text=generic_task)
        logger.info(str(result))

    # --- 8. Task-conditioned ---
    logger.info("\n--- Evaluation (task-conditioned) ---")
    logger.info(f"{'Dataset':15s} | {'MRR@10':>8s} | {'NDCG@10':>8s} | {'Recall@100':>10s}")
    logger.info("-" * 55)

    for name, ds in eval_datasets.items():
        result = evaluate_retriever(retriever, ds)
        logger.info(str(result))


if __name__ == "__main__":
    main()
