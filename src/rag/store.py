"""ChromaDB vector store for document embeddings.

Wraps ChromaDB with an interface matched to our Chunk/Embedder types.
Handles persistence, upsert, and similarity search.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import chromadb

from src.data.schemas import Chunk, RetrievedChunk
from src.rag.embedder import Embedder


@dataclass
class VectorStore:
    """ChromaDB-backed vector store.

    Usage:
        store = VectorStore(embedder=Embedder())
        store.add_chunks(chunks)
        results = store.query("How does X work?", top_k=5)
    """

    embedder: Embedder
    persist_directory: str = "./data/embeddings/chromadb"
    collection_name: str = "rag_docs"
    distance_metric: str = "cosine"
    _client: chromadb.ClientAPI | None = field(default=None, init=False, repr=False)
    _collection: chromadb.Collection | None = field(default=None, init=False, repr=False)

    @property
    def client(self) -> chromadb.ClientAPI:
        if self._client is None:
            Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=self.persist_directory)
        return self._client

    @property
    def collection(self) -> chromadb.Collection:
        if self._collection is None:
            self._collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": self.distance_metric},
            )
        return self._collection

    def add_chunks(self, chunks: list[Chunk], batch_size: int = 50) -> None:
        """Embed and store chunks in the vector store.

        Args:
            chunks: List of Chunk objects to add.
            batch_size: Number of chunks to embed at once.
        """
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            texts = [c.text for c in batch]
            ids = [c.chunk_id for c in batch]
            embeddings = self.embedder.embed_documents(texts)
            metadatas = [
                {
                    "source_path": c.metadata.source_path,
                    "doc_type": c.metadata.doc_type.value,
                    "title": c.metadata.title or "",
                    "language": c.metadata.language or "",
                }
                for c in batch
            ]

            self.collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
            )
            print(f"  Indexed {min(i + batch_size, len(chunks))}/{len(chunks)} chunks")

    def query(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        """Search for chunks most relevant to the query.

        Args:
            query: Natural language query.
            top_k: Number of results to return.

        Returns:
            List of RetrievedChunk objects sorted by relevance.
        """
        query_embedding = self.embedder.embed_query(query)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances", "embeddings"],
        )

        retrieved = []
        for idx in range(len(results["ids"][0])):
            from src.data.schemas import DocumentMeta, DocumentType

            meta = results["metadatas"][0][idx]
            chunk = Chunk(
                chunk_id=results["ids"][0][idx],
                text=results["documents"][0][idx],
                metadata=DocumentMeta(
                    source_path=meta["source_path"],
                    doc_type=DocumentType(meta["doc_type"]),
                    title=meta.get("title"),
                    language=meta.get("language") or None,
                ),
                embedding=results["embeddings"][0][idx] if results.get("embeddings") else None,
            )
            # ChromaDB returns distances; convert to similarity score
            distance = results["distances"][0][idx]
            score = 1.0 - distance  # cosine distance → similarity

            retrieved.append(RetrievedChunk(chunk=chunk, score=score))

        return retrieved

    def count(self) -> int:
        """Return the number of chunks in the store."""
        return self.collection.count()

    def reset(self) -> None:
        """Delete all data in the collection."""
        self.client.delete_collection(self.collection_name)
        self._collection = None
