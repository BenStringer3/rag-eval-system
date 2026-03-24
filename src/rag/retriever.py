"""Retrieval layer.

Dense Chroma search; optional hybrid BM25 + reciprocal rank fusion (RRF).
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.data.schemas import RetrievedChunk
from src.rag.store import VectorStore

if TYPE_CHECKING:
    from src.rag.bm25_index import Bm25Index


def reciprocal_rank_fusion(
    dense_ids: list[str],
    sparse_ids: list[str],
    k: int,
) -> list[tuple[str, float]]:
    """RRF scores; higher is better. Ties broken lexicographically by chunk_id."""
    scores: dict[str, float] = {}
    for rank, cid in enumerate(dense_ids, start=1):
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    for rank, cid in enumerate(sparse_ids, start=1):
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    ordered = sorted(scores.keys(), key=lambda x: (-scores[x], x))
    return [(cid, scores[cid]) for cid in ordered]


@dataclass
class RetrieveOutcome:
    chunks: list[RetrievedChunk]
    timings_ms: dict[str, float]


@dataclass
class Retriever:
    """Retrieve relevant chunks for a query.

    Phase 1: Pure cosine similarity via ChromaDB.
    Hybrid: BM25 sidecar + RRF fusion, then text-hash dedup.
    """

    store: VectorStore
    top_k: int = 5
    score_threshold: float | None = None
    hybrid_enabled: bool = False
    bm25_index: Bm25Index | None = None
    candidate_k: int = 20
    rrf_k: int = 60

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        return self.retrieve_detailed(query).chunks

    def retrieve_detailed(self, query: str) -> RetrieveOutcome:
        if self.hybrid_enabled:
            return self._retrieve_hybrid(query)
        return self._retrieve_dense_only(query)

    def _retrieve_dense_only(self, query: str) -> RetrieveOutcome:
        t0 = time.perf_counter()
        results = self.store.query(query, top_k=self.top_k * 2)
        dense_ms = (time.perf_counter() - t0) * 1000

        t1 = time.perf_counter()
        results = self._deduplicate(results)
        if self.score_threshold is not None:
            results = [r for r in results if r.score >= self.score_threshold]
        results = results[: self.top_k]
        fuse_ms = (time.perf_counter() - t1) * 1000

        return RetrieveOutcome(
            chunks=results,
            timings_ms={"dense": dense_ms, "bm25": 0.0, "fuse": fuse_ms},
        )

    def _retrieve_hybrid(self, query: str) -> RetrieveOutcome:
        if self.bm25_index is None or not self.bm25_index.built:
            raise RuntimeError(
                "Hybrid retrieval is enabled but the BM25 index is missing or empty. "
                "Run corpus ingest so data/embeddings (Chroma + bm25) are built."
            )

        t0 = time.perf_counter()
        dense = self.store.query(query, top_k=self.candidate_k)
        dense_ms = (time.perf_counter() - t0) * 1000

        t1 = time.perf_counter()
        sparse_pairs = self.bm25_index.search(query, self.candidate_k)
        sparse_ids = [cid for cid, _ in sparse_pairs]
        bm25_ms = (time.perf_counter() - t1) * 1000

        t2 = time.perf_counter()
        dense_ids = [rc.chunk.chunk_id for rc in dense]
        fused = reciprocal_rank_fusion(dense_ids, sparse_ids, self.rrf_k)

        by_dense = {rc.chunk.chunk_id: rc for rc in dense}
        missing = [cid for cid, _ in fused if cid not in by_dense]
        fetched = self.store.get_chunks_by_ids(missing)

        ordered: list[RetrievedChunk] = []
        for cid, rrf_score in fused:
            if cid in by_dense:
                base = by_dense[cid]
            elif cid in fetched:
                base = fetched[cid]
            else:
                continue
            ordered.append(base.model_copy(update={"score": rrf_score}))

        ordered = self._deduplicate(ordered)
        # RRF scores are not comparable to cosine similarity; skip threshold in hybrid mode.
        ordered = ordered[: self.top_k]
        fuse_ms = (time.perf_counter() - t2) * 1000

        return RetrieveOutcome(
            chunks=ordered,
            timings_ms={"dense": dense_ms, "bm25": bm25_ms, "fuse": fuse_ms},
        )

    def retrieve_as_context(self, query: str) -> list[str]:
        """Convenience method: return just the text of retrieved chunks.

        This is the format expected by DeepEval's retrieval_context parameter.
        """
        results = self.retrieve(query)
        return [r.chunk.text for r in results]

    @staticmethod
    def _deduplicate(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """Remove chunks with identical text regardless of source file.

        Keeps the first fused/dense-ordered copy (typically highest RRF rank).
        """
        seen: set[str] = set()
        unique: list[RetrievedChunk] = []
        for rc in chunks:
            text_hash = hashlib.sha256(rc.chunk.text.encode()).hexdigest()
            if text_hash not in seen:
                seen.add(text_hash)
                unique.append(rc)
        return unique
