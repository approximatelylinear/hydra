"""ColBERT retriever: frozen base encoder + hypernet-generated FiLM for multi-vector retrieval."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from hydra.hypernet.colbert_head_generator import ColBERTHeadGenerator
from hydra.hypernet.encoder import TaskCardEncoder


class ColBERTBaseEncoder(nn.Module):
    """Wraps a frozen ColBERTv2 (or compatible) model for per-token embeddings.

    Falls back to any HuggingFace transformer + linear projection if
    colbert-ir/colbertv2.0 is not available.
    """

    def __init__(self, model_name: str = "colbert-ir/colbertv2.0", embed_dim: int = 128):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.embed_dim = embed_dim

        # ColBERTv2 has a linear projection layer; if the hidden size differs
        # from embed_dim, add one.
        hidden_size = self.model.config.hidden_size
        if hidden_size != embed_dim:
            self.projection = nn.Linear(hidden_size, embed_dim, bias=False)
        else:
            self.projection = nn.Identity()

        # Freeze everything
        for p in self.model.parameters():
            p.requires_grad = False
        if isinstance(self.projection, nn.Linear):
            for p in self.projection.parameters():
                p.requires_grad = False

    def forward(
        self,
        texts: list[str],
        max_length: int = 180,
        is_query: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode texts into per-token embeddings.

        Args:
            texts: input texts
            max_length: max token length
            is_query: if True, prepend [Q] marker (ColBERT convention)

        Returns:
            (token_embs, mask): token_embs is (batch, seq, embed_dim),
                                mask is (batch, seq) boolean attention mask
        """
        if is_query:
            texts = [f"[Q] {t}" for t in texts]

        device = next(self.model.parameters()).device
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = self.model(**encoded)

        token_embs = outputs.last_hidden_state  # (batch, seq, hidden)
        token_embs = self.projection(token_embs)  # (batch, seq, embed_dim)
        token_embs = F.normalize(token_embs, p=2, dim=-1)
        mask = encoded["attention_mask"].bool()

        return token_embs, mask


def batched_maxsim(
    q_embs: torch.Tensor,
    d_embs: torch.Tensor,
    q_mask: torch.Tensor,
    d_mask: torch.Tensor,
) -> torch.Tensor:
    """Differentiable MaxSim scoring for ColBERT late interaction.

    score(q, d) = Σ_i max_j cos(q_i, d_j)  (summed over valid query tokens)

    Args:
        q_embs: (batch, q_len, dim) L2-normalized query token embeddings
        d_embs: (batch, d_len, dim) L2-normalized doc token embeddings
        q_mask: (batch, q_len) boolean mask for query tokens
        d_mask: (batch, d_len) boolean mask for doc tokens

    Returns:
        (batch,) MaxSim scores
    """
    # (batch, q_len, d_len) cosine similarity matrix (already L2-normed)
    sim = torch.bmm(q_embs, d_embs.transpose(1, 2))

    # Mask out padding doc tokens: set to -inf so they don't win the max
    d_mask_expanded = d_mask.unsqueeze(1).expand_as(sim)  # (batch, q_len, d_len)
    sim = sim.masked_fill(~d_mask_expanded, float("-inf"))

    # Max over doc tokens for each query token
    max_sim, _ = sim.max(dim=2)  # (batch, q_len)

    # Zero out padded query positions, then sum
    max_sim = max_sim.masked_fill(~q_mask, 0.0)
    scores = max_sim.sum(dim=1)  # (batch,)

    return scores


class ColBERTRetriever(nn.Module):
    """End-to-end ColBERT retriever conditioned on a task card.

    Pipeline:
        task_card_text -> TaskCardEncoder -> cond_vector
        cond_vector -> ColBERTHeadGenerator -> head_params (FiLM gamma/beta/alpha)
        query/doc -> frozen ColBERT base -> per-token embeddings
        per-token embeddings -> apply_head(head_params) -> task-adapted token embeddings
        scoring via MaxSim late interaction
    """

    def __init__(
        self,
        base_model: str = "colbert-ir/colbertv2.0",
        task_encoder_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        cond_dim: int = 256,
        embed_dim: int = 128,
        rank: int = 32,
    ):
        super().__init__()
        self.base_encoder = ColBERTBaseEncoder(model_name=base_model, embed_dim=embed_dim)
        self.task_encoder = TaskCardEncoder(base_model=task_encoder_model, cond_dim=cond_dim)
        self.head_gen = ColBERTHeadGenerator(cond_dim=cond_dim, embed_dim=embed_dim, rank=rank)

    def compile_task(self, task_text: str) -> dict[str, torch.Tensor]:
        """Compile a task card into FiLM modulation parameters (cache-friendly)."""
        cond = self.task_encoder([task_text])  # (1, cond_dim)
        return self.head_gen(cond)

    def encode_multi_vector(
        self,
        texts: list[str],
        head_params: dict[str, torch.Tensor],
        max_length: int = 180,
        is_query: bool = False,
        batch_size: int = 64,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode texts into task-adapted per-token embeddings.

        Args:
            texts: input texts
            head_params: output of compile_task()
            max_length: max token length
            is_query: whether these are queries (adds [Q] prefix)
            batch_size: encoding batch size

        Returns:
            (token_embs, mask): token_embs is (total, max_seq, embed_dim),
                                mask is (total, max_seq) boolean
        """
        all_embs = []
        all_masks = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            with torch.no_grad():
                token_embs, mask = self.base_encoder(
                    batch_texts, max_length=max_length, is_query=is_query
                )
            # Clone to escape inference_mode tensors
            token_embs = token_embs.clone()
            # Apply FiLM modulation
            token_embs = self.head_gen.apply_head(token_embs, head_params)
            all_embs.append(token_embs)
            all_masks.append(mask)

        # Pad to same seq length across batches
        max_seq = max(e.size(1) for e in all_embs)
        padded_embs = []
        padded_masks = []
        for embs, mask in zip(all_embs, all_masks):
            seq_len = embs.size(1)
            if seq_len < max_seq:
                pad_size = max_seq - seq_len
                embs = F.pad(embs, (0, 0, 0, pad_size))
                mask = F.pad(mask, (0, pad_size), value=False)
            padded_embs.append(embs)
            padded_masks.append(mask)

        return torch.cat(padded_embs, dim=0), torch.cat(padded_masks, dim=0)

    def score_pairs(
        self,
        queries: list[str],
        docs: list[str],
        task_text: str,
        max_query_len: int = 32,
        max_doc_len: int = 180,
    ) -> torch.Tensor:
        """Score query-doc pairs under a task via MaxSim. Returns (n_pairs,) scores."""
        head_params = self.compile_task(task_text)
        q_embs, q_mask = self.encode_multi_vector(
            queries, head_params, max_length=max_query_len, is_query=True
        )
        d_embs, d_mask = self.encode_multi_vector(
            docs, head_params, max_length=max_doc_len, is_query=False
        )
        return batched_maxsim(q_embs, d_embs, q_mask, d_mask)
