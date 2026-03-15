# Experiment Roadmap

What we've tried, what worked, what to try next.

## What We Know

### Architecture: 20 experiments, 3 key findings

**1. Trainable shared A/B matrices collapse at scale.**
Quick experiments (3 epochs, 2 datasets) consistently show +0.002-0.003 task deltas with FiLM + shared low-rank residual. But every extensive experiment (15 epochs, 5 datasets) collapsed — generic NDCG dropped 5-17% below baseline. Three mitigation attempts all failed:

| Fix attempted | Extensive avg NDCG | Baseline | Verdict |
|---|---|---|---|
| No fix (rank 64) | 0.3430 | 0.3834 | -10.5% |
| Alpha cap 0.3 | 0.3548 | 0.3834 | -7.5% |
| Weight decay 0.1 on A/B | 0.3182 | 0.3834 | -17.0% (worse!) |
| Warmup-freeze (3 ep) | 0.3706 | 0.3834 | -3.3% |

Root cause: shared A/B matrices accumulate a permanent bias during extended training that degrades base embedding quality regardless of regularization.

**2. Direct FiLM (no shared matrices) is stable at scale.**
Removing A/B entirely and applying `gamma * x + beta` directly to embeddings solved collapse. Extensive experiment: avg NDCG 0.3829 vs baseline 0.3834 (within noise). But task differentiation is near-zero at scale — the FiLM generator converges to task-agnostic modulation because the pairwise loss doesn't incentivize differentiation.

**3. ColBERT multi-vector + direct FiLM shows the strongest task differentiation.**
ColBERT with direct FiLM achieved avg task delta +0.0043 across 3 datasets — the best result in any experiment. Per-token modulation gives FiLM more surface area to influence the embedding space. Also: FiLM must be applied to both queries and docs at eval to match training; omitting docs caused catastrophic collapse (NDCG ~0.005).

### Training signal: 5 experiments, 2 key findings

**4. Cross-task contrastive loss on conditioning vectors is ineffective.**
With only 2-5 tasks, the contrastive loss trivially saturates (cosine similarity → -1.0 within a few hundred steps). The conditioning vectors become orthogonal but the FiLM generator still produces similar modulation — orthogonal inputs don't guarantee functionally different outputs. Tested lambda=0.1 and 0.5; neither improved task delta over baseline. Not worth pursuing unless task count increases significantly (20+).

**5. Negative task conditioning is the breakthrough for task differentiation.**
With 20% probability, applying a random wrong task card and penalizing when it scores well directly teaches the model that task cards matter. Results:

| Mode | avg NDCG | task_vs_generic | Key finding |
|---|---|---|---|
| Quick (2 datasets, 3 ep) | 0.5115 | +0.0036 | Best quick task delta for bi-encoder |
| **Extensive (5 datasets, 15 ep)** | **0.3838** | **+0.0023** | **First positive delta on ALL 5 datasets at scale** |

The extensive result is a milestone: avg NDCG matches baseline (0.3838 vs 0.3834) with **positive task delta on every dataset**. Previous best extensive delta was -0.0001. Config: `neg_task_prob=0.2, neg_task_margin=0.1, neg_task_lambda=0.5`.

Per-dataset extensive results:
| Dataset | Baseline | Generic | Task | Delta |
|---|---|---|---|---|
| scifact | 0.6451 | 0.6418 | 0.6452 | +0.0033 |
| fiqa | 0.3687 | 0.3693 | 0.3723 | +0.0030 |
| nfcorpus | 0.3168 | 0.3140 | 0.3162 | +0.0022 |
| arguana | 0.3699 | 0.3653 | 0.3684 | +0.0031 |
| scidocs | 0.2164 | 0.2166 | 0.2167 | +0.0001 |

### Architecture decisions locked in

| Decision | Chosen | Why |
|---|---|---|
| FiLM variant | Direct (no A/B) | Only variant that survives extensive training |
| Task card encoder | Multi-exemplar attention pooling | +0.0032 task delta, best in quick experiments |
| FiLM generator depth | 2 hidden layers (cond_dim * 2) | Wider (cond_dim * 4) reduced differentiation |
| TaskCardEncoder projection | 2-layer | 4-layer showed no improvement |
| Alpha cap | 0.3 | 0.5 didn't help; 0.1 too constrained |
| Bi-encoder base model | all-MiniLM-L6-v2 (frozen) | Prototyping model, Nomic later |
| ColBERT base model | colbertv2.0 (frozen) | Standard ColBERT |

### Training signal decisions locked in

| Decision | Chosen | Why |
|---|---|---|
| Negative task conditioning | prob=0.2, margin=0.1, lambda=0.5 | First method to produce positive task delta at scale |
| Cross-task contrastive loss | Not used | Trivially saturates with few tasks, no benefit |

---

## Proposed Experiments

### Tier 1: High confidence, likely impactful

#### 1C. Synthetic task augmentation via query clustering
**Hypothesis:** 5 training tasks isn't enough diversity for the hypernet. Clustering queries into sub-tasks multiplies diversity without new data. With neg_task conditioning already working, more tasks = more negative task signal = stronger differentiation.

**Evidence:** Neg task conditioning produces +0.0023 at scale with 5 tasks. More sub-tasks would provide richer negative examples.

**Implementation:** K-means on query embeddings → 5-10 sub-tasks per dataset → each gets its own task card with sub-task exemplars.

**Files:** `hydra/data/preference_pairs.py` (or new `hydra/data/task_augmentation.py`)
**Expected impact:** +0.003-0.008 task delta. Multiplicative with neg_task conditioning.
**Risk:** Medium. Sub-task boundaries may be arbitrary; exemplar quality matters.

#### 1D. Apply neg_task conditioning to ColBERT trainer
**Hypothesis:** ColBERT already shows +0.0043 task delta without neg_task conditioning. Adding it should push differentiation even higher since ColBERT has more modulation surface area (per-token FiLM).

**Evidence:** Neg_task went from -0.0001 to +0.0023 for bi-encoder. ColBERT starts at +0.0043 — combining could yield +0.006-0.010.

**Implementation:** Port the neg_task_prob/margin/lambda logic from `trainer.py` to `colbert_trainer.py`.

**Files:** `hydra/training/colbert_trainer.py`
**Expected impact:** +0.003-0.005 additional task delta on top of ColBERT's existing +0.0043.
**Risk:** Low. Same mechanism, just applied to ColBERT.

### Tier 2: Moderate confidence, worth trying

#### 2A. ColBERT query-only FiLM (train + eval aligned)
**Hypothesis:** Applying FiLM only to queries during both training and eval would be faster at inference (encode corpus once, serve multiple tasks) and may produce better differentiation since the model focuses its modulation budget on queries.

**Evidence:** The current ColBERT experiment applies FiLM to both queries and docs during training. This is correct but means the corpus must be re-encoded per task at inference.

**Implementation:** Modify `train_colbert_hypernet` to skip `apply_head` for doc encoding. Evaluate with `apply_head_to_docs=False`.

**Files:** `hydra/training/colbert_trainer.py`
**Expected impact:** Neutral-to-positive on quality (+/- 0.002), significant inference speedup.
**Risk:** Low. If quality drops, revert.

#### 2B. Larger ColBERT experiment (more epochs, more datasets)
**Hypothesis:** The ColBERT direct FiLM experiment only ran 10 epochs on 3 datasets. It may benefit from more training data and longer training, especially since direct FiLM was proven stable at 15 epochs for the bi-encoder.

**Evidence:** ColBERT loss was still decreasing at epoch 10 (0.44 → 0.36). Bi-encoder direct FiLM was stable through 15 epochs at full scale.

**Implementation:** Run `run_colbert_experiment.py` with 5 datasets, 15 epochs, all teachers.

**Files:** `scripts/run_colbert_experiment.py`
**Expected impact:** +0.005-0.015 absolute NDCG (from more training), task delta uncertain.
**Risk:** Low. Direct FiLM doesn't collapse.

#### 2C. Separate query/doc FiLM parameters
**Hypothesis:** Asymmetric tasks (e.g., arguana: argument→counter-argument) need different modulation for queries vs docs.

**Evidence:** Previously abandoned because the bi-encoder evaluator calls `encode()` without a role parameter. But the ColBERT evaluator already separates query and doc encoding (`is_query=True/False`), making this cleanly implementable.

**Implementation:** Generate `gamma_q, beta_q, gamma_d, beta_d` from conditioning. Route by `is_query` flag in `apply_head`.

**Files:** `hydra/hypernet/colbert_head_generator.py`, `hydra/student/colbert_retriever.py`
**Expected impact:** +0.002-0.005 task delta, especially on asymmetric tasks (arguana).
**Risk:** Medium. Doubles FiLM params. Previously showed no benefit with A/B bottleneck, but direct FiLM + ColBERT may be different. Hard to test on bi-encoder due to evaluator API constraint.

#### 2D. Learning rate schedule (cosine decay)
**Hypothesis:** Constant LR may be suboptimal. Early training needs higher LR for fast feature learning; late training benefits from lower LR for fine-tuning the FiLM generator.

**Evidence:** Loss trajectories show rapid early improvement then plateau. No LR schedule has been tried.

**Implementation:** Add cosine annealing with warmup to `TrainConfig`.

**Files:** `hydra/training/trainer.py`
**Expected impact:** +0.001-0.003 avg NDCG.
**Risk:** Very low. Standard technique.

#### 2E. Higher neg_task_prob or larger margin
**Hypothesis:** The current neg_task_prob=0.2 may be conservative. Higher probability or larger margin could push more differentiation, especially on scidocs which barely responded (+0.0001).

**Evidence:** prob=0.5 showed more balanced deltas across datasets in quick experiments but lower peak scifact delta.

**Implementation:** Try prob=0.3/0.4, margin=0.2/0.3.

**Files:** `hydra/training/trainer.py` (config only)
**Expected impact:** +0.001-0.003 additional task delta.
**Risk:** Very low. Quick experiments only.

### Tier 3: Speculative, lower confidence

#### 3A. Structurally diverse training tasks
**Hypothesis:** Current tasks (scifact, fiqa, nfcorpus, arguana, scidocs) are all "find relevant passage" with different domains. Adding fundamentally different retrieval tasks (code search, bitext mining, duplicate detection) would force the hypernet to learn genuinely different embedding geometries.

**Evidence:** The design doc proposes this as Phase 4. No experiments yet. The concern is that these tasks may require different base encoders (MiniLM isn't great at code).

**Implementation:** Add MTEB tasks with non-passage-retrieval relevance. Requires new data loaders.

**Expected impact:** High if it works (+0.01+ task delta), but implementation is heavy.
**Risk:** High. Data format differences, potential base model limitations, increased training time.

#### 3B. Per-task loss weighting
**Hypothesis:** Tasks with more training data (arguana: 1406 queries → 22496 pairs) may dominate the gradient over smaller tasks (scifact: 300 → 4800). Inverse-frequency weighting could help underrepresented tasks.

**Evidence:** Task-grouped batching already ensures per-batch task purity, but tasks with more data produce more batches. scidocs (+0.0001) and nfcorpus (+0.0022) responded least to neg_task — possibly gradient-starved.

**Implementation:** Weight loss by `1/n_pairs_for_task` per batch.

**Expected impact:** +0.001-0.002 on smaller tasks.
**Risk:** Low, but impact may be small.

#### 3C. Temperature tuning
**Hypothesis:** Temperature 0.05 may be too aggressive for ColBERT MaxSim scores, which have different magnitude than dot-product scores.

**Evidence:** ColBERT loss plateaued at 0.36 (higher than bi-encoder's 0.17). MaxSim scores are sums over tokens, so they're much larger than single-vector dot products.

**Implementation:** Try temperature 0.01, 0.02, 0.1 for ColBERT.

**Expected impact:** +0.005-0.010 NDCG if current temperature is mismatched.
**Risk:** Very low. Quick experiments only.

---

## Recommended Execution Order

```
1. [1D] Neg task conditioning → ColBERT       ← proven technique, apply to stronger architecture
2. [2D] LR schedule                            ← cheap, universally helpful
3. [1C] Synthetic task augmentation            ← multiplies training diversity
4. [2E] Tune neg_task hyperparams              ← quick, may help scidocs/nfcorpus
5. [3C] ColBERT temperature tuning             ← quick win if mismatched
6. [2B] Larger ColBERT experiment              ← validate at scale with neg_task
7. [2A] ColBERT query-only FiLM               ← inference optimization
8. [2C] Separate query/doc FiLM               ← asymmetric task improvement
9. [3A] Structurally diverse tasks             ← scaling, needs new data
```

Priority: Apply neg_task conditioning to ColBERT (1D) since it's proven and ColBERT already has the best architecture for task differentiation. Then try LR schedule and task augmentation.
