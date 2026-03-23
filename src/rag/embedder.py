"""Local embeddings via LM Studio (OpenAI-compatible /v1/embeddings).

Nomic-style models use task prefixes ("search_query: " / "search_document: ")
for best retrieval quality.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from openai import OpenAI


@dataclass
class Embedder:
    """Generate embeddings using LM Studio's OpenAI-compatible API.

    Usage:
        client = openai_client(LMStudioSettings.from_config(cfg))
        embedder = Embedder(client=client, model="...")
        doc_vectors = embedder.embed_documents(["Hello world", "Foo bar"])
        query_vector = embedder.embed_query("What is hello?")
    """

    client: OpenAI
    model: str
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
        resp = self.client.embeddings.create(model=self.model, input=[text])
        return list(resp.data[0].embedding)
