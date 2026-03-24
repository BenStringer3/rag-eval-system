"""Lexical BM25 index over chunk texts (sidecar to Chroma dense retrieval)."""

from __future__ import annotations

import pickle
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

from src.data.schemas import Chunk

_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)


def tokenize(text: str) -> list[str]:
    """Lowercase word/number tokens; no stemming (good enough for technical docs)."""
    return _TOKEN_RE.findall(text.lower())


class Bm25Index:
    """BM25 over ingested chunks; persisted as tokenized corpus + ids (rebuild Okapi on load)."""

    def __init__(self) -> None:
        self._chunk_ids: list[str] = []
        self._tokenized: list[list[str]] = []
        self._bm25: BM25Okapi | None = None

    @property
    def built(self) -> bool:
        return self._bm25 is not None and len(self._chunk_ids) > 0

    def build(self, chunks: list[Chunk]) -> None:
        """Fit BM25 on the given chunks (same order and ids as Chroma upsert)."""
        self._chunk_ids = [c.chunk_id for c in chunks]
        self._tokenized = [tokenize(c.text) for c in chunks]
        self._bm25 = BM25Okapi(self._tokenized) if self._tokenized else None

    def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        """Return top_k (chunk_id, bm25_score) by descending relevance."""
        if not self._bm25 or not self._chunk_ids:
            return []
        q = tokenize(query)
        scores = self._bm25.get_scores(q)
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [(self._chunk_ids[i], float(scores[i])) for i in order]

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"v": 1, "chunk_ids": self._chunk_ids, "tokenized": self._tokenized}
        with path.open("wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: Path) -> Bm25Index:
        path = Path(path)
        with path.open("rb") as f:
            data = pickle.load(f)
        if data.get("v") != 1:
            raise ValueError(f"Unsupported BM25 index version: {data.get('v')}")
        inst = cls()
        inst._chunk_ids = data["chunk_ids"]
        inst._tokenized = data["tokenized"]
        inst._bm25 = BM25Okapi(inst._tokenized) if inst._tokenized else None
        return inst
