"""Local embedding via Ollama.

Uses nomic-embed-text by default. The model requires task-specific
prefixes ("search_query: " for queries, "search_document: " for docs)
to produce optimal embeddings.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import ollama


@dataclass
class Embedder:
    """Generate embeddings using a local Ollama model.

    Usage:
        embedder = Embedder()
        doc_vectors = embedder.embed_documents(["Hello world", "Foo bar"])
        query_vector = embedder.embed_query("What is hello?")
    """

    model: str = "nomic-embed-text"
    query_prefix: str = "search_query: "
    document_prefix: str = "search_document: "
    _dimensions: int | None = field(default=None, init=False, repr=False)

    @property
    def dimensions(self) -> int:
        """Lazily detect embedding dimensionality."""
        if self._dimensions is None:
            test = self._embed_single("test")
            self._dimensions = len(test)
        return self._dimensions

    def embed_query(self, query: str) -> list[float]:
        """Embed a search query with the query prefix."""
        return self._embed_single(f"{self.query_prefix}{query}")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of document chunks with the document prefix."""
        return [self._embed_single(f"{self.document_prefix}{text}") for text in texts]

    def embed_raw(self, texts: list[str]) -> list[list[float]]:
        """Embed without any prefix (for visualization/analysis)."""
        return [self._embed_single(text) for text in texts]

    def _embed_single(self, text: str) -> list[float]:
        """Call Ollama embedding API for a single text."""
        response = ollama.embed(model=self.model, input=text)
        return response["embeddings"][0]
