"""Training loop for ColBERT multi-vector hypernet."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from hydra.data.preference_pairs import PreferencePair
from hydra.student.colbert_retriever import ColBERTRetriever, batched_maxsim
from hydra.training.pairwise_loss import pairwise_margin_loss_from_scores
from hydra.training.trainer import (
    PreferencePairDataset,
    TaskGroupedSampler,
    collate_preferences,
)

logger = logging.getLogger(__name__)


@dataclass
class ColBERTTrainConfig:
    lr: float = 1e-4
    batch_size: int = 16
    epochs: int = 10
    warmup_steps: int = 100
    temperature: float = 0.05
    max_query_len: int = 32
    max_doc_len: int = 180
    checkpoint_dir: str = "checkpoints"
    log_every: int = 50
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def train_colbert_hypernet(
    retriever: ColBERTRetriever,
    pairs: list[PreferencePair],
    task_cards: dict[str, str],
    config: ColBERTTrainConfig | None = None,
) -> ColBERTRetriever:
    """Train the ColBERT hypernet to match teacher pairwise preferences.

    Same structure as train_hypernet but uses multi-vector encoding + MaxSim.
    """
    config = config or ColBERTTrainConfig()
    device = torch.device(config.device)
    retriever = retriever.to(device)

    # Only optimize hypernet params (not base encoder)
    trainable = [
        {"params": retriever.task_encoder.projection.parameters()},
        {"params": retriever.task_encoder.attn_pool.parameters()},
        {"params": retriever.head_gen.param_gen.parameters()},
    ]
    optimizer = AdamW(trainable, lr=config.lr, weight_decay=0.01)

    dataset = PreferencePairDataset(pairs)
    sampler = TaskGroupedSampler(pairs, batch_size=config.batch_size)

    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collate_preferences,
        num_workers=0,
    )

    retriever.train()
    global_step = 0

    for epoch in range(config.epochs):
        epoch_loss = 0.0
        sampler.set_epoch(epoch)

        for batch in loader:
            task_name: str = batch["task_name"]
            task_text = task_cards.get(task_name) or task_name

            head_params = retriever.compile_task(task_text)

            # Encode queries and docs as multi-vector
            q_embs, q_mask = retriever.encode_multi_vector(
                batch["queries"],
                head_params,
                max_length=config.max_query_len,
                is_query=True,
            )
            pos_embs, pos_mask = retriever.encode_multi_vector(
                batch["docs_pos"],
                head_params,
                max_length=config.max_doc_len,
                is_query=False,
            )
            neg_embs, neg_mask = retriever.encode_multi_vector(
                batch["docs_neg"],
                head_params,
                max_length=config.max_doc_len,
                is_query=False,
            )

            # MaxSim scores
            pos_scores = batched_maxsim(q_embs, pos_embs, q_mask, pos_mask)
            neg_scores = batched_maxsim(q_embs, neg_embs, q_mask, neg_mask)

            margins = batch["margins"].to(device)

            loss = pairwise_margin_loss_from_scores(
                pos_scores, neg_scores, margins, temperature=config.temperature
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(retriever.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            global_step += 1

            if global_step % config.log_every == 0:
                logger.info(f"Step {global_step} | Task: {task_name} | Loss: {loss.item():.4f}")

        avg_loss = epoch_loss / max(len(loader), 1)
        logger.info(f"Epoch {epoch + 1}/{config.epochs} | Avg Loss: {avg_loss:.4f}")

    # Save checkpoint
    ckpt_dir = Path(config.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "task_encoder_proj": retriever.task_encoder.projection.state_dict(),
            "head_gen": retriever.head_gen.state_dict(),
        },
        ckpt_dir / "colbert_hypernet.pt",
    )
    logger.info(f"Saved checkpoint to {ckpt_dir / 'colbert_hypernet.pt'}")

    return retriever
