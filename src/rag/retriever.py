"""Retrieval layer.

Currently a thin wrapper around VectorStore.query(). This is where
future upgrades plug in: hybrid search, reranking, HyDE, etc.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.data.schemas import RetrievedChunk
from src.rag.store import VectorStore


@dataclass
class Retriever:
    """Retrieve relevant chunks for a query.

    Phase 1: Pure cosine similarity via ChromaDB.
    Phase 3+: Hybrid search, reranking, HyDE.
    """

    store: VectorStore
    top_k: int = 5
    score_threshold: float | None = None

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        """Find the most relevant chunks for a query.

        Args:
            query: Natural language question.

        Returns:
            Ranked list of RetrievedChunk objects.
        """
        results = self.store.query(query, top_k=self.top_k)

        if self.score_threshold is not None:
            results = [r for r in results if r.score >= self.score_threshold]

        return results

    def retrieve_as_context(self, query: str) -> list[str]:
        """Convenience method: return just the text of retrieved chunks.

        This is the format expected by DeepEval's retrieval_context parameter.
        """
        results = self.retrieve(query)
        return [r.chunk.text for r in results]
