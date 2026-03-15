"""Pairwise preference distillation loss for hypernet training."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def pairwise_margin_loss(
    q_embs: torch.Tensor,
    pos_embs: torch.Tensor,
    neg_embs: torch.Tensor,
    margins: torch.Tensor | None = None,
    temperature: float = 0.05,
) -> torch.Tensor:
    """Pairwise loss: student should score pos higher than neg.

    Uses a margin-scaled softmax formulation rather than raw score matching,
    so we distill orderings not calibration.

    Args:
        q_embs: (batch, dim) query embeddings
        pos_embs: (batch, dim) positive doc embeddings
        neg_embs: (batch, dim) negative doc embeddings
        margins: (batch,) optional teacher margin weights
        temperature: softmax temperature
    """
    pos_scores = (q_embs * pos_embs).sum(dim=-1) / temperature
    neg_scores = (q_embs * neg_embs).sum(dim=-1) / temperature

    # Log-sigmoid loss: -log(sigmoid(pos - neg))
    loss = -F.logsigmoid(pos_scores - neg_scores)

    if margins is not None:
        # Weight by teacher confidence (higher margin = more confident preference)
        weights = torch.clamp(margins, min=0.1)
        weights = weights / weights.mean()
        loss = loss * weights

    return loss.mean()


def pairwise_margin_loss_from_scores(
    pos_scores: torch.Tensor,
    neg_scores: torch.Tensor,
    margins: torch.Tensor | None = None,
    temperature: float = 0.05,
) -> torch.Tensor:
    """Pairwise loss from pre-computed scalar scores (e.g. MaxSim).

    Same formulation as pairwise_margin_loss but takes scores directly
    instead of computing dot products from embeddings.

    Args:
        pos_scores: (batch,) positive pair scores
        neg_scores: (batch,) negative pair scores
        margins: (batch,) optional teacher margin weights
        temperature: softmax temperature
    """
    loss = -F.logsigmoid((pos_scores - neg_scores) / temperature)

    if margins is not None:
        weights = torch.clamp(margins, min=0.1)
        weights = weights / weights.mean()
        loss = loss * weights

    return loss.mean()


def in_batch_contrastive_loss(
    q_embs: torch.Tensor,
    d_embs: torch.Tensor,
    temperature: float = 0.05,
) -> torch.Tensor:
    """In-batch contrastive loss (InfoNCE).

    Positives are on the diagonal; all other docs in the batch are negatives.
    """
    # (batch, batch) similarity matrix
    sim = torch.mm(q_embs, d_embs.t()) / temperature
    labels = torch.arange(sim.size(0), device=sim.device)
    return F.cross_entropy(sim, labels)
