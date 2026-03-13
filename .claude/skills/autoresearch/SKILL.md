---
name: autoresearch
description: Semi-autonomous iterative experiment loop for improving hydra's hypernet-conditioned retrieval. Run quick training experiments to find signal, log results, keep improvements, revert failures. Use when the user wants to run experiments, try architectural changes, tune hyperparameters, or improve NDCG/MRR metrics.
disable-model-invocation: true
argument-hint: [track] [idea]
allowed-tools: Read, Edit, Write, Grep, Glob, Bash, Agent
---

# Autoresearch — Hydra Experiment Loop

You are running the hydra autoresearch loop: a semi-autonomous process for iteratively improving the hypernet-conditioned retrieval system. The goal is to find training configurations and architectural changes that improve task-conditioned retrieval metrics (NDCG@10, MRR@10) on BEIR benchmarks.

## Arguments

- **Track** (first argument): `$ARGUMENTS[0]` — one of `architecture`, `training`, `hyperparams`, or `auto` (pick best track)
- **Idea** (remaining arguments): `$ARGUMENTS[1:]` — optional specific idea to try. If empty, read the relevant files and propose ideas.

## Tracks

### Track 1: Architecture
**Goal**: Improve the hypernet architecture to better differentiate tasks.
**Files you may modify**: `hydra/hypernet/head_generator.py`, `hydra/hypernet/encoder.py`, `hydra/hypernet/task_card.py`, `hydra/student/conditioned_retriever.py`
**Ideas** (see `docs/design-next-steps.md` for details):
- FiLM conditioning (generate scale/shift instead of full weight matrices)
- Multi-exemplar encoder (attention-pool exemplar embeddings instead of single text encoding)
- Separate query/doc FiLM params for asymmetric tasks
- Deeper projection in TaskCardEncoder (4-6 layers instead of 2)
- Multi-head conditioning (separate query_cond, doc_cond, global_cond)
- Adjusting low-rank residual rank (currently 64)

### Track 2: Training Signal
**Goal**: Improve training diversity and loss functions.
**Files you may modify**: `hydra/training/trainer.py`, `hydra/training/pairwise_loss.py`, `hydra/data/preference_pairs.py`, `hydra/data/beir_loader.py`
**Ideas**:
- Synthetic task augmentation (cluster queries → sub-tasks with their own exemplars)
- Cross-task contrastive loss (push conditioning vectors apart)
- Negative task conditioning (apply wrong task card + penalize)
- Hard negative ratio tuning
- Different loss functions (InfoNCE, triplet, listwise)
- Per-task loss weighting

### Track 3: Hyperparameters
**Goal**: Find better training configurations.
**Files you may modify**: `scripts/run_minimal_experiment.py`, `hydra/training/trainer.py`
**Ideas**:
- Learning rate schedule (warmup, cosine decay)
- Batch size (currently 32)
- Epochs (currently 15)
- Conditioning dimension (currently 256)
- Temperature (currently 0.05)
- Candidates per query (currently 100)
- Pairs per query (currently 16)
- Teacher ensemble weights (currently BM25=1, dense=1, Jina=3)
- Weight decay
- Gradient clipping threshold
- Adding/removing training datasets

## Files you MUST NOT modify

- `hydra/eval/evaluator.py` — the evaluation harness is fixed
- `hydra/eval/metrics.py` — the metric implementations are fixed
- `hydra/teachers/` — teacher implementations are fixed (they provide the ground truth signal)

## Quick vs Extensive Experiments

The loop alternates between two modes:

### Quick experiments (~2-5 min)
- Use 2-3 datasets (scifact + fiqa for fast signal — they show the most hypernet benefit)
- 3-5 epochs
- Fewer candidates per query (50)
- Fewer pairs per query (8)
- Skip the Jina reranker (use BM25 + dense only) for faster preference generation
- **Purpose**: Rapid iteration to find promising directions

### Extensive experiments (~15-30 min)
- Full 5-dataset suite (scifact, fiqa, nfcorpus, arguana, scidocs)
- 15 epochs
- Full candidate/pair counts (100/16)
- Full teacher ensemble including Jina reranker
- **Purpose**: Validate that quick-experiment wins hold up

**Schedule**: Run 3-5 quick experiments, then 1 extensive experiment to validate the best change so far.

## Setup

### Step 1: Check state

```bash
git status
git log --oneline -5
```

Verify you are on an `autoresearch/*` branch. If not, create one:
```bash
git checkout -b autoresearch/<tag>
```

Check if `results.tsv` exists. If not, create it with header and run baseline.

### Step 2: Baseline (if first run)

Create a quick experiment script variant for fast iteration. Run baseline eval:

```bash
cd /home/mjberends/Code/hydra
uv run python -c "
from hydra.data.beir_loader import load_beir_dataset
from hydra.eval.evaluator import evaluate_baseline
BASE_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
for name in ['scifact', 'fiqa']:
    ds = load_beir_dataset(name)
    r = evaluate_baseline(BASE_MODEL, ds)
    print(r)
"
```

Create `results.tsv`:
```
commit	track	mode	scifact_ndcg10	fiqa_ndcg10	avg_ndcg10	task_vs_generic	status	description
<hash>	baseline	quick	<score>	<score>	<avg>	n/a	keep	frozen MiniLM baseline (no hypernet)
```

### Step 3: Read track files

Read ALL files listed for the chosen track. Understand the current implementation before making changes.

### Step 4: Choose and implement an idea

Pick ONE focused idea. Keep changes small and targeted — one variable per experiment.

**Simplicity criterion**: simpler is better. A small NDCG improvement that adds ugly complexity is not worth it. Removing code and getting equal or better results is a win.

### Step 5: Commit

```bash
git add <changed files>
git commit -m "<short description>"
```

### Step 6: Run experiment

**Quick mode** — create/modify a lightweight script or run inline:

```python
# Quick experiment: 2 datasets, 3 epochs, no Jina reranker
import torch
from hydra.data.beir_loader import load_beir_dataset
from hydra.data.preference_pairs import generate_preference_pairs
from hydra.eval.evaluator import evaluate_retriever
from hydra.student.conditioned_retriever import ConditionedRetriever
from hydra.teachers.bm25 import BM25Teacher
from hydra.teachers.dense import DenseTeacher
from hydra.teachers.ensemble import EnsembleTeacher
from hydra.training.trainer import TrainConfig, train_hypernet

device = "cuda" if torch.cuda.is_available() else "cpu"
datasets = {n: load_beir_dataset(n) for n in ["scifact", "fiqa"]}

# Generate preferences (fast: BM25 + dense only)
dense = DenseTeacher(model_name="sentence-transformers/all-MiniLM-L6-v2")
all_pairs = []
for name, ds in datasets.items():
    bm25 = BM25Teacher()
    bm25.index(ds.corpus_texts)
    dense.index(ds.corpus_texts)
    ens = EnsembleTeacher(teachers=[bm25, dense], weights=[1.0, 1.0])
    pairs = generate_preference_pairs(
        queries=list(ds.queries.values()), corpus_texts=ds.corpus_texts,
        teacher=ens, candidates_per_query=50, pairs_per_query=8,
        hard_negative_ratio=0.5, task_name=name,
    )
    all_pairs.extend(pairs)

# Train
retriever = ConditionedRetriever(base_model="sentence-transformers/all-MiniLM-L6-v2", cond_dim=256)
task_cards = {n: ds.task_card.to_text() if ds.task_card else n for n, ds in datasets.items()}
config = TrainConfig(lr=1e-4, batch_size=32, epochs=3, device=device)
retriever = train_hypernet(retriever, all_pairs, task_cards, config)

# Evaluate: generic vs task-conditioned
generic = "Retrieve relevant documents for diverse information needs"
for name, ds in datasets.items():
    r_generic = evaluate_retriever(retriever, ds, task_text=generic)
    r_task = evaluate_retriever(retriever, ds)
    delta = r_task.ndcg_10 - r_generic.ndcg_10
    print(f"{name:12s} | generic={r_generic.ndcg_10:.4f} | task={r_task.ndcg_10:.4f} | delta={delta:+.4f}")
```

For **extensive mode**, use the full `scripts/run_minimal_experiment.py`.

### Step 7: Record and decide

Extract NDCG@10 scores and the task-conditioned vs generic gap. Append to `results.tsv`.

**Key metrics** (in priority order):
1. **avg_ndcg10**: Average NDCG@10 across eval datasets (higher is better)
2. **task_vs_generic**: Average (task-conditioned NDCG@10 - generic NDCG@10) across datasets. This measures whether task cards actually help. Positive = good.

**Decision rules:**
- avg_ndcg10 improved over best previous → `keep`
- avg_ndcg10 improved by > 0.02 → `review` (flag for human)
- task_vs_generic turned positive or increased significantly → strong `keep`
- avg_ndcg10 equal or worse → `discard` and `git reset --hard HEAD~1`
- Experiment crashed → log as `crash`, revert, move on

### Step 8: Report and continue

After each experiment, briefly report:
- What you changed (1 line)
- Scores: avg NDCG@10 and task_vs_generic delta
- Decision (keep/discard/review/crash)
- Time taken

Then continue to the next experiment unless:
1. **`review` status** — describe the change and wait for human approval
2. **3+ consecutive failures** — pause and describe what you've tried
3. **Track switching** — summarize progress and propose next track
4. **Architectural question** — if considering a fundamental change (new loss function, different base model), check first

Otherwise, keep going. The human may be away and expects progress when they return.

## Metrics Reference

| Metric | What it measures | Target |
|---|---|---|
| NDCG@10 | Ranking quality of top 10 results | Higher is better; baseline ~0.56 scifact, ~0.28 fiqa |
| MRR@10 | Reciprocal rank of first relevant doc | Higher is better |
| Recall@100 | Fraction of relevant docs in top 100 | Higher is better |
| task_vs_generic | NDCG@10(task-conditioned) - NDCG@10(generic) | Positive = task cards help |

## Current Best Results (reference)

| Dataset  | Baseline | Generic | Task-conditioned |
|----------|----------|---------|------------------|
| scifact  | 0.5579   | 0.6447  | 0.6415           |
| fiqa     | 0.2781   | 0.3707  | 0.3704           |
| nfcorpus | 0.3168   | 0.3172  | 0.3136           |
| arguana  | 0.3768   | 0.3666  | 0.3692           |
| scidocs  | 0.2164   | 0.2171  | 0.2158           |

**Key observation**: The hypernet learns a good general residual (+8-9 NDCG on scifact/fiqa) but task-conditioned barely beats generic. The main goal is to widen the task_vs_generic gap while maintaining or improving overall NDCG.

## Timing Expectations

- Quick experiment (2 datasets, 3 epochs, no Jina): ~2-5 min
- Extensive experiment (5 datasets, 15 epochs, Jina reranker): ~20-40 min
- Preference generation (BM25 + dense): ~1-2 min per dataset
- Preference generation (with Jina reranker): ~5-10 min per dataset
- Evaluation per dataset: ~10-30s
