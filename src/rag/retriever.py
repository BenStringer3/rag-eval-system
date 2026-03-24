"""Retrieval layer.

Currently a thin wrapper around VectorStore.query(). This is where
future upgrades plug in: hybrid search, reranking, HyDE, etc.
"""

from __future__ import annotations

import hashlib
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

        Over-fetches from the store (2× top_k) to compensate for duplicates
        removed by dedup, then returns up to top_k unique chunks.

        Args:
            query: Natural language question.

        Returns:
            Ranked list of RetrievedChunk objects (deduplicated).
        """
        results = self.store.query(query, top_k=self.top_k * 2)
        results = self._deduplicate(results)

        if self.score_threshold is not None:
            results = [r for r in results if r.score >= self.score_threshold]

        return results[: self.top_k]

    def retrieve_as_context(self, query: str) -> list[str]:
        """Convenience method: return just the text of retrieved chunks.

        This is the format expected by DeepEval's retrieval_context parameter.
        """
        results = self.retrieve(query)
        return [r.chunk.text for r in results]

    @staticmethod
    def _deduplicate(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """Remove chunks with identical text regardless of source file.

        Keeps the highest-scored copy when the same text appears from
        multiple source documents (e.g. repeated log snippets).
        """
        seen: set[str] = set()
        unique: list[RetrievedChunk] = []
        for rc in chunks:
            text_hash = hashlib.sha256(rc.chunk.text.encode()).hexdigest()
            if text_hash not in seen:
                seen.add(text_hash)
                unique.append(rc)
        return unique
