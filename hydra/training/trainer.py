"""Training loop for the hypernet + student retriever."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from hydra.data.preference_pairs import PreferencePair
from hydra.student.conditioned_retriever import ConditionedRetriever
from hydra.training.pairwise_loss import pairwise_margin_loss

logger = logging.getLogger(__name__)


class PreferencePairDataset(Dataset):
    """Wraps preference pairs for DataLoader."""

    def __init__(self, pairs: list[PreferencePair]):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        p = self.pairs[idx]
        return {"query": p.query, "doc_pos": p.doc_pos, "doc_neg": p.doc_neg, "margin": p.margin}


def collate_preferences(batch: list[dict]) -> dict:
    return {
        "queries": [b["query"] for b in batch],
        "docs_pos": [b["doc_pos"] for b in batch],
        "docs_neg": [b["doc_neg"] for b in batch],
        "margins": torch.tensor([b["margin"] for b in batch], dtype=torch.float32),
    }


@dataclass
class TrainConfig:
    lr: float = 1e-4
    batch_size: int = 32
    epochs: int = 10
    warmup_steps: int = 100
    temperature: float = 0.05
    checkpoint_dir: str = "checkpoints"
    log_every: int = 50
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def train_hypernet(
    retriever: ConditionedRetriever,
    pairs: list[PreferencePair],
    task_text: str,
    config: TrainConfig | None = None,
) -> ConditionedRetriever:
    """Train the hypernet to match teacher pairwise preferences.

    Only the hypernet parameters (task_encoder.projection + head_gen) are trained.
    Base encoder stays frozen.
    """
    config = config or TrainConfig()
    device = torch.device(config.device)
    retriever = retriever.to(device)

    # Only optimize hypernet params (not base encoder)
    trainable = [
        {"params": retriever.task_encoder.projection.parameters()},
        {"params": retriever.head_gen.parameters()},
    ]
    optimizer = AdamW(trainable, lr=config.lr, weight_decay=0.01)

    dataset = PreferencePairDataset(pairs)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_preferences,
        num_workers=0,
    )

    retriever.train()
    global_step = 0

    for epoch in range(config.epochs):
        epoch_loss = 0.0

        for batch in loader:
            # Compile task -> head params
            head_params = retriever.compile_task(task_text)

            # Encode queries and docs through frozen base + generated head
            q_embs = retriever.encode(batch["queries"], head_params)
            pos_embs = retriever.encode(batch["docs_pos"], head_params)
            neg_embs = retriever.encode(batch["docs_neg"], head_params)

            margins = batch["margins"].to(device)

            loss = pairwise_margin_loss(
                q_embs, pos_embs, neg_embs, margins, temperature=config.temperature
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(retriever.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            global_step += 1

            if global_step % config.log_every == 0:
                logger.info(f"Step {global_step} | Loss: {loss.item():.4f}")

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
        ckpt_dir / "hypernet.pt",
    )
    logger.info(f"Saved checkpoint to {ckpt_dir / 'hypernet.pt'}")

    return retriever
