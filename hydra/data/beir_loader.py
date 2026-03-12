"""Load BEIR datasets and convert to a common format for teacher/student training."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from beir import util as beir_util
from beir.datasets.data_loader import GenericDataLoader

from hydra.hypernet.task_card import TaskCard

# Task card templates for BEIR datasets we'll use
BEIR_TASK_CARDS: dict[str, dict] = {
    "trec-covid": {
        "name": "trec-covid",
        "description": "Find scientific articles about COVID-19 that answer biomedical questions",
        "domain": "biomedical",
        "query_type": "factoid",
    },
    "nfcorpus": {
        "name": "nfcorpus",
        "description": "Find nutrition and health documents relevant to medical queries",
        "domain": "health/nutrition",
        "query_type": "factoid",
    },
    "fiqa": {
        "name": "fiqa",
        "description": "Find financial opinion and advice passages that answer financial questions",
        "domain": "finance",
        "query_type": "opinion",
    },
    "scidocs": {
        "name": "scidocs",
        "description": "Find related scientific papers given a paper as query",
        "domain": "scientific",
        "query_type": "similarity",
    },
    "scifact": {
        "name": "scifact",
        "description": "Find scientific abstracts that support or refute scientific claims",
        "domain": "scientific",
        "query_type": "claim_verification",
    },
}


@dataclass
class RetrievalDataset:
    """Common format for a retrieval dataset."""

    name: str
    corpus: dict[str, str]  # doc_id -> text
    queries: dict[str, str]  # query_id -> text
    qrels: dict[str, dict[str, int]]  # query_id -> {doc_id: relevance}
    task_card: TaskCard | None = None

    # Materialized lists for indexed access
    corpus_ids: list[str] = field(default_factory=list)
    corpus_texts: list[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.corpus_ids:
            self.corpus_ids = list(self.corpus.keys())
            self.corpus_texts = [self.corpus[did] for did in self.corpus_ids]

    @property
    def doc_id_to_idx(self) -> dict[str, int]:
        return {did: i for i, did in enumerate(self.corpus_ids)}


def load_beir_dataset(
    dataset_name: str,
    split: str = "test",
    data_dir: str | Path = "data/raw/beir",
) -> RetrievalDataset:
    """Download (if needed) and load a BEIR dataset."""
    data_dir = Path(data_dir)
    dataset_path = data_dir / dataset_name

    if not dataset_path.exists():
        url = (
            f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"
        )
        beir_util.download_and_unzip(url, str(data_dir))

    corpus, queries, qrels = GenericDataLoader(str(dataset_path)).load(split=split)

    # Flatten corpus: combine title + text
    flat_corpus = {}
    for doc_id, doc in corpus.items():
        title = doc.get("title", "")
        text = doc.get("text", "")
        flat_corpus[doc_id] = f"{title} {text}".strip() if title else text

    # Build task card from template or generic
    card_data = BEIR_TASK_CARDS.get(
        dataset_name,
        {
            "name": dataset_name,
            "description": f"Retrieve relevant documents for {dataset_name} queries",
        },
    )

    # Add example queries and docs to the card for richer conditioning
    sample_queries = list(queries.values())[:20]
    # Grab a few short doc snippets as exemplars
    sample_docs = [text[:200] for text in list(flat_corpus.values())[:10]]
    card = TaskCard(query_examples=sample_queries, doc_examples=sample_docs, **card_data)

    return RetrievalDataset(
        name=dataset_name,
        corpus=flat_corpus,
        queries=queries,
        qrels=qrels,
        task_card=card,
    )
