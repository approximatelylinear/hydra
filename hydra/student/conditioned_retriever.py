"""Conditioned retriever: frozen base encoder + hypernet-generated residual adaptation."""

from __future__ import annotations

import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer

from hydra.hypernet.encoder import TaskCardEncoder
from hydra.hypernet.head_generator import ProjectionHeadGenerator


class ConditionedRetriever(nn.Module):
    """End-to-end retriever conditioned on a task card.

    Pipeline:
        task_card_text -> TaskCardEncoder -> cond_vector
        cond_vector -> ProjectionHeadGenerator -> head_params
        query/doc -> frozen base encoder -> base_embedding
        base_embedding -> apply_head(head_params) -> task-specific embedding
    """

    def __init__(
        self,
        base_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        cond_dim: int = 256,
    ):
        super().__init__()
        self.base_encoder = SentenceTransformer(base_model)
        for p in self.base_encoder.parameters():
            p.requires_grad = False

        embed_dim = self.base_encoder.get_sentence_embedding_dimension()
        assert embed_dim is not None, f"Could not determine embedding dim for {base_model}"

        self.task_encoder = TaskCardEncoder(base_model=base_model, cond_dim=cond_dim)
        self.head_gen = ProjectionHeadGenerator(cond_dim=cond_dim, embed_dim=embed_dim)

    def compile_task(self, task_text: str) -> dict[str, torch.Tensor]:
        """Compile a task card into projection head parameters (cache-friendly)."""
        cond = self.task_encoder([task_text])  # (1, cond_dim)
        return self.head_gen(cond)

    def encode(
        self,
        texts: list[str],
        head_params: dict[str, torch.Tensor],
        batch_size: int = 256,
    ) -> torch.Tensor:
        """Encode texts using base encoder + task-specific projection head."""
        with torch.no_grad():
            base_embs = self.base_encoder.encode(
                texts, batch_size=batch_size, convert_to_tensor=True, normalize_embeddings=True
            ).clone()  # clone to escape inference_mode tensors from sentence-transformers
        return ProjectionHeadGenerator.apply_head(base_embs, head_params)

    def score_pairs(
        self,
        queries: list[str],
        docs: list[str],
        task_text: str,
    ) -> torch.Tensor:
        """Score query-doc pairs under a task. Returns (n_pairs,) similarity scores."""
        head_params = self.compile_task(task_text)
        q_embs = self.encode(queries, head_params)
        d_embs = self.encode(docs, head_params)
        return (q_embs * d_embs).sum(dim=-1)
