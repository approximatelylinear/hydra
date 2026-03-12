"""Task card: structured description of a retrieval task for hypernet conditioning."""

from __future__ import annotations

from pydantic import BaseModel


class TaskCard(BaseModel):
    """Describes a retrieval task so the hypernet can infer what 'relevance' means.

    This is the retrieval analog of "doc -> LoRA" — enough info to generate
    a task-specific embedding head without fine-tuning.
    """

    name: str
    description: str  # e.g. "find relevant invoices for compliance auditing"
    query_examples: list[str] = []  # 3-20 example queries
    doc_examples: list[str] = []  # 3-10 exemplar relevant docs/snippets
    domain: str = ""  # e.g. "finance", "legal", "engineering"
    query_type: str = ""  # e.g. "factoid", "navigational", "keyword"

    def to_text(self, max_query_examples: int = 20, max_doc_examples: int = 10) -> str:
        """Flatten to a single text block for encoding.

        Includes as many exemplars as possible — these give the hypernet
        distributional signal about the task, not just a description.
        """
        parts = [
            f"Task: {self.name}",
            f"Description: {self.description}",
        ]
        if self.domain:
            parts.append(f"Domain: {self.domain}")
        if self.query_type:
            parts.append(f"Query type: {self.query_type}")
        if self.query_examples:
            examples = self.query_examples[:max_query_examples]
            parts.append("Example queries:")
            for i, q in enumerate(examples, 1):
                parts.append(f"  {i}. {q}")
        if self.doc_examples:
            examples = self.doc_examples[:max_doc_examples]
            parts.append("Example documents:")
            for i, d in enumerate(examples, 1):
                parts.append(f"  {i}. {d}")
        return "\n".join(parts)
