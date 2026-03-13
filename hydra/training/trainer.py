"""Training loop for the hypernet + student retriever."""

from __future__ import annotations

import logging
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, Sampler

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
        return {
            "query": p.query,
            "doc_pos": p.doc_pos,
            "doc_neg": p.doc_neg,
            "margin": p.margin,
            "task_name": p.task_name,
        }


class TaskGroupedSampler(Sampler):
    """Yields batches where all examples come from the same task.

    Each batch is a single-task block. Tasks are interleaved across batches
    so the hypernet sees different task cards throughout training.
    """

    def __init__(
        self,
        pairs: list[PreferencePair],
        batch_size: int,
        shuffle: bool = True,
        seed: int = 42,
    ):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

        # Group indices by task
        self.task_to_indices: dict[str, list[int]] = defaultdict(list)
        for i, p in enumerate(pairs):
            self.task_to_indices[p.task_name].append(i)

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)

        # Build per-task batches
        all_batches: list[list[int]] = []
        for task_name, indices in self.task_to_indices.items():
            task_indices = indices.copy()
            if self.shuffle:
                rng.shuffle(task_indices)
            # Chunk into batches
            for i in range(0, len(task_indices), self.batch_size):
                all_batches.append(task_indices[i : i + self.batch_size])

        # Shuffle the batch order so tasks are interleaved
        if self.shuffle:
            rng.shuffle(all_batches)

        # Yield individual indices — DataLoader will group them back via batch_sampler
        for batch in all_batches:
            yield batch

    def __len__(self):
        total = 0
        for indices in self.task_to_indices.values():
            total += (len(indices) + self.batch_size - 1) // self.batch_size
        return total


def collate_preferences(batch: list[dict]) -> dict:
    return {
        "queries": [b["query"] for b in batch],
        "docs_pos": [b["doc_pos"] for b in batch],
        "docs_neg": [b["doc_neg"] for b in batch],
        "margins": torch.tensor([b["margin"] for b in batch], dtype=torch.float32),
        "task_name": batch[0]["task_name"],  # all same within a batch
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
    task_cards: dict[str, str],
    config: TrainConfig | None = None,
) -> ConditionedRetriever:
    """Train the hypernet to match teacher pairwise preferences.

    Each batch contains pairs from a single task, and the hypernet compiles
    that task's card into a projection head for the batch.

    Args:
        retriever: The conditioned retriever (base encoder frozen, hypernet trainable).
        pairs: Preference pairs with task_name set.
        task_cards: Mapping from task_name to task card text.
        config: Training configuration.
    """
    config = config or TrainConfig()
    device = torch.device(config.device)
    retriever = retriever.to(device)

    # Only optimize hypernet params (not base encoder, not frozen A/B buffers)
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

            # Compile this task's card -> head params
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
        ckpt_dir / "hypernet.pt",
    )
    logger.info(f"Saved checkpoint to {ckpt_dir / 'hypernet.pt'}")

    return retriever
