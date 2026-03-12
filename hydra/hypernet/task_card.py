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

    def to_text(self) -> str:
        """Flatten to a single text block for encoding."""
        parts = [
            f"Task: {self.name}",
            f"Description: {self.description}",
        ]
        if self.domain:
            parts.append(f"Domain: {self.domain}")
        if self.query_type:
            parts.append(f"Query type: {self.query_type}")
        if self.query_examples:
            parts.append("Example queries: " + " | ".join(self.query_examples[:5]))
        if self.doc_examples:
            parts.append("Example docs: " + " | ".join(self.doc_examples[:3]))
        return "\n".join(parts)
