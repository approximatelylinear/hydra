# Hydra

Hypernet-conditioned retrieval: encode a wide distribution of embedding tasks without per-task fine-tuning.

## Architecture

- **Teachers** (`hydra/teachers/`): BM25, dense bi-encoder, ensemble (RRF fusion). Provide supervision signal.
- **Hypernet** (`hydra/hypernet/`): TaskCard → conditioning vector → generated projection head weights.
- **Student** (`hydra/student/`): Frozen base encoder + hypernet-generated projection head.
- **Training** (`hydra/training/`): Pairwise preference distillation (orderings, not raw scores).
- **Eval** (`hydra/eval/`): Per-task MRR, NDCG, Recall on BEIR datasets.

## Quick start

```bash
pip install -e ".[dev]"
python scripts/run_minimal_experiment.py
```

## Key design decisions

- Base encoder is frozen (all-MiniLM-L6-v2 for prototyping, Nomic later)
- Hypernet output: 384→256 projection + LayerNorm params (small, fast)
- Teacher signal: pairwise preferences (robust to miscalibration)
- Task conditioning via "task cards" (description + exemplars)

## Conventions

- PyTorch + HuggingFace Transformers
- Use BEIR datasets for evaluation (scifact, fiqa, nfcorpus for fast iteration)
- Configs via dataclasses, not YAML files
- Ruff for linting
