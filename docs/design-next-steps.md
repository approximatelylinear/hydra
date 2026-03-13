# Hydra: Design Doc — Improving Task Conditioning

## Status Quo

The residual hypernet architecture is working: +8-9 NDCG@10 over the frozen baseline on scifact/fiqa, with no regression on other tasks. However, **task-conditioned evaluation barely beats generic conditioning**. The hypernet is learning a good general residual but isn't differentiating per-task.

### Root Cause

The conditioning signal is too weak. All task cards pass through the same frozen MiniLM encoder, which maps retrieval instructions to a similar embedding region. The hypernet sees nearly identical conditioning vectors for different tasks and has no reason to produce different residuals.

### Current Results (Jina Reranker v3 teacher, 3x weighted RRF)

| Dataset  | Baseline | Generic | Task-conditioned |
|----------|----------|---------|------------------|
| scifact  | 0.5579   | 0.6447  | 0.6415           |
| fiqa     | 0.2781   | 0.3707  | 0.3704           |
| nfcorpus | 0.3168   | 0.3172  | 0.3136           |
| arguana  | 0.3768   | 0.3666  | 0.3692           |
| scidocs  | 0.2164   | 0.2171  | 0.2158           |

---

## Part 1: Training Signal Diversity

### 1.1 Exemplar-Based Task Encoding

**Problem:** Task card descriptions ("Find scientific abstracts that support or refute claims") are too semantically similar across tasks when embedded by MiniLM.

**Solution:** Encode example queries and docs independently, then pool. The *distribution* of exemplars is far more distinctive per-task than any single description string.

```
Current:   "Task: scifact\nDescription: ...\nExample queries: q1 | q2 | q3" → MiniLM → one vector
Proposed:  [q1, q2, ..., q20] → MiniLM → 20 vectors → attention pooling → conditioning
           [d1, d2, ..., d10] → MiniLM → 10 vectors → attention pooling → conditioning
```

This is analogous to prototypical networks — the task is represented by a prototype computed from its support set, not from a text label.

### 1.2 Synthetic Task Augmentation

**Problem:** 5 training tasks isn't enough diversity for the hypernet to learn that different cards should produce different behavior.

**Solution:** Generate synthetic "tasks" by slicing existing datasets:
- **By query cluster:** K-means on query embeddings → each cluster is a "sub-task" with its own exemplars
- **By difficulty:** Split by teacher confidence (high-agreement vs contentious pairs)
- **By document type:** Cluster corpus docs → create tasks like "retrieve long-form scientific text" vs "retrieve short factual snippets"

A dataset with 300 queries can become 10-20 synthetic sub-tasks, each with 15-30 exemplars. This multiplies task diversity without needing new data.

### 1.3 Cross-Task Contrastive Regularization

**Problem:** No explicit training signal pushes different task cards toward different residuals. The hypernet can collapse to one good residual for everything.

**Solution:** Add a contrastive term to the training loss:

```
L_total = L_pairwise + lambda * L_contrastive

L_contrastive = -log(exp(sim(z_i, z_i)) / sum_j(exp(sim(z_i, z_j))))
```

where `z_i` is the conditioning vector for task `i`. This pushes conditioning vectors apart in embedding space, which forces the head generator to produce distinct residuals.

Alternatively, apply contrastive loss on the generated head params directly (the A, B matrices or the alpha values) — this is more direct but may be harder to optimize.

### 1.4 Negative Task Conditioning

**Problem:** The hypernet never sees evidence that the *wrong* task card hurts performance.

**Solution:** During training, with some probability (e.g., 10-20%), apply a random *wrong* task card to a batch. Add a loss term that penalizes the model when it performs well with the wrong card (or equivalently, rewards it when the correct card outperforms the wrong one):

```
L_neg = max(0, margin - (score_correct_card - score_wrong_card))
```

This teaches the hypernet that conditioning actually matters — it can't just ignore the task card.

### 1.5 Structurally Diverse Tasks

**Problem:** Current tasks are mostly "find relevant passage for query" — same fundamental operation, different domains.

**Solution:** Add tasks with fundamentally different notions of relevance:
- **Bitext mining** (translation pairs) — relevance = same meaning, different language
- **Citation prediction** — relevance = would cite this paper
- **Code search** — relevance = this code implements this description
- **Duplicate detection** — relevance = asks the same question differently
- **Fact verification** — relevance = supports or refutes (not just topically related)

These force the hypernet to learn genuinely different embedding geometries, not just domain-specific tweaks.

---

## Part 2: Architectural Improvements

### 2.1 Multi-Exemplar Encoder (replaces current TaskCardEncoder)

**Problem:** One text → one embedding is a severe information bottleneck for task conditioning.

**Proposed architecture:**

```
Example queries [q1..q20]  → frozen MiniLM → [e1..e20] → learned attention pooling → q_proto (384-dim)
Example docs    [d1..d10]  → frozen MiniLM → [e1..e10] → learned attention pooling → d_proto (384-dim)
Task description           → frozen MiniLM → desc_emb (384-dim)

[q_proto; d_proto; desc_emb] → MLP → conditioning vector (cond_dim)
```

The attention pooling is a small learned module:
```python
class AttentionPooling(nn.Module):
    def __init__(self, dim):
        self.attn = nn.Linear(dim, 1)

    def forward(self, embeddings):  # (n_exemplars, dim)
        weights = softmax(self.attn(embeddings), dim=0)
        return (weights * embeddings).sum(dim=0)
```

Conditioning is now informed by the *distribution* of queries and docs for the task, not just a text description.

### 2.2 FiLM Conditioning (replaces full weight generation)

**Problem:** Generating the full low-rank residual (A: embed_dim x rank, B: rank x embed_dim, bias) from a single conditioning vector is a lot of parameters to predict well. The hypernet has to learn a mapping from conditioning space to weight space, which is hard with few tasks.

**Proposed:** Use Feature-wise Linear Modulation (FiLM). Keep a single *learned* shared residual path, and generate only scale/shift parameters per task:

```
Current:
  conditioning → generate A, B, bias, alpha (all per-task)
  residual = x @ A @ B + bias
  output = x + alpha * residual

Proposed:
  Shared (learned, not generated):
    A_shared: (embed_dim, rank)  — trained via backprop
    B_shared: (rank, embed_dim)  — trained via backprop

  Generated per-task (from conditioning):
    gamma: (embed_dim,)  — FiLM scale
    beta:  (embed_dim,)  — FiLM shift
    alpha: scalar         — mixing weight

  residual = x @ A_shared @ B_shared
  modulated = gamma * residual + beta
  output = normalize(x + alpha * modulated)
```

**Why this helps:**
- The shared A, B capture common structure across all tasks (the "most useful residual direction")
- FiLM params are tiny (2 * embed_dim + 1 per task vs embed_dim * rank * 2 + embed_dim + 1)
- Much easier to train — the hypernet only needs to learn how to modulate, not how to construct an entire linear transform
- FiLM is proven in visual QA, style transfer, and meta-learning settings

### 2.3 Separate Query/Doc Adaptation

**Problem:** The same residual is applied to both queries and docs. For asymmetric tasks (e.g., arguana: query is an argument, relevant doc is a *counter*-argument), the query and doc spaces should shift differently.

**Proposed:** Generate separate FiLM params for query vs doc encoding:

```
conditioning → gamma_q, beta_q, gamma_d, beta_d, alpha

query:  output_q = normalize(q + alpha * (gamma_q * (q @ A @ B) + beta_q))
doc:    output_d = normalize(d + alpha * (gamma_d * (d @ A @ B) + beta_d))
```

The shared A, B are still shared between query and doc paths (they capture the general residual direction). Only the modulation differs.

This adds minimal parameters (2x the FiLM params) but lets the hypernet learn that "for argument retrieval, shift queries toward the claim space and docs toward the rebuttal space."

### 2.4 Multi-Head Conditioning

**Problem:** A single conditioning vector is a bottleneck — it must encode everything about what "relevance" means.

**Proposed:** Split conditioning into multiple heads with different roles:

```
exemplar encoder → [query_proto, doc_proto, desc_emb]
                         ↓             ↓           ↓
                   query_cond     doc_cond    global_cond

query_cond  → FiLM params for query encoder
doc_cond    → FiLM params for doc encoder
global_cond → alpha (mixing weight) + any shared params
```

Each conditioning head has a clear semantic role, making the hypernet's job easier.

### 2.5 Dedicated Task Card Encoder (instead of shared MiniLM)

**Problem:** Using the same MiniLM for task cards and for query/doc encoding means the conditioning space is constrained by the retrieval embedding geometry. Retrieval instructions cluster together.

**Options:**
- **Separate small encoder:** Train a lightweight encoder (e.g., 2-layer transformer) specifically for task cards. It doesn't need to be good at retrieval — it needs to produce *discriminative* task representations.
- **Different frozen encoder:** Use a model trained for NLI or instruction-following (e.g., an instruction-tuned model) that better separates different task descriptions.
- **Deeper projection:** Keep MiniLM but add a much deeper projection (4-6 layers instead of 2) with dropout to give the conditioning space more flexibility to reorganize.

The simplest version: just make the projection deeper.

---

## Recommended Implementation Order

### Phase 1: Quick wins (highest impact, lowest effort)
1. **Multi-exemplar encoder (2.1)** — Replace single-text encoding with attention-pooled exemplars. This directly addresses the weakest link.
2. **FiLM conditioning (2.2)** — Simplify what the hypernet has to generate. Fewer params to predict = easier to train with few tasks.

### Phase 2: Training signal improvements
3. **Synthetic task augmentation (1.2)** — Multiply task diversity via clustering. No new data needed.
4. **Cross-task contrastive loss (1.3)** — Explicit pressure to differentiate tasks.

### Phase 3: Asymmetric and multi-head
5. **Separate query/doc adaptation (2.3)** — Unlocks asymmetric tasks like arguana.
6. **Negative task conditioning (1.4)** — Reinforces that task cards matter.
7. **Multi-head conditioning (2.4)** — Full system with role-specific conditioning.

### Phase 4: Scaling
8. **Structurally diverse tasks (1.5)** — Add fundamentally different retrieval tasks.
9. **Dedicated task card encoder (2.5)** — Only needed once the rest of the pipeline is working and the bottleneck shifts to conditioning quality.
