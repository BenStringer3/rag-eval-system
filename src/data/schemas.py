"""Data schemas for the RAG evaluation system.

These Pydantic models define the shape of evaluation datasets,
pipeline outputs, and metric results. They serve as the contract
between the RAG pipeline, evaluation layer, and visualization tools.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Document & Chunk models
# ---------------------------------------------------------------------------

class DocumentType(str, Enum):
    MARKDOWN = "markdown"
    CODE = "code"
    MERMAID = "mermaid"
    TEXT = "text"


class DocumentMeta(BaseModel):
    """Metadata attached to a source document."""

    source_path: str
    doc_type: DocumentType
    title: str | None = None
    language: str | None = None  # e.g. "python", "typescript" for code files


class Chunk(BaseModel):
    """A single chunk produced by the chunker."""

    chunk_id: str
    text: str
    metadata: DocumentMeta
    start_char: int | None = None
    end_char: int | None = None
    embedding: list[float] | None = None  # Populated after embedding


# ---------------------------------------------------------------------------
# Evaluation dataset models
# ---------------------------------------------------------------------------

class EvalSample(BaseModel):
    """A single evaluation sample (ground truth Q&A pair).

    This is the unit of evaluation. At minimum, you need `query` and
    `expected_answer`. The optional fields unlock more metrics:
      - `expected_context` enables contextual recall
      - `metadata` enables per-category analysis
    """

    id: str
    query: str
    expected_answer: str
    expected_context: list[str] | None = None  # Ground truth context snippets
    metadata: dict[str, Any] = Field(default_factory=dict)
    # e.g. {"category": "code", "difficulty": "hard", "doc_type": "python"}


class EvalDataset(BaseModel):
    """A collection of evaluation samples."""

    name: str
    description: str = ""
    samples: list[EvalSample]
    version: str = "1.0.0"

    @property
    def size(self) -> int:
        return len(self.samples)

    def filter_by_metadata(self, key: str, value: Any) -> EvalDataset:
        """Return a filtered subset based on metadata field."""
        filtered = [s for s in self.samples if s.metadata.get(key) == value]
        return EvalDataset(
            name=f"{self.name}_filtered_{key}={value}",
            description=f"Filtered from {self.name} where {key}={value}",
            samples=filtered,
            version=self.version,
        )


# ---------------------------------------------------------------------------
# Pipeline output models
# ---------------------------------------------------------------------------

class RetrievedChunk(BaseModel):
    """A chunk returned by the retriever with its relevance score."""

    chunk: Chunk
    score: float  # Similarity score (higher = more relevant)


class RAGResult(BaseModel):
    """The full output of a single RAG pipeline invocation."""

    query: str
    answer: str
    retrieved_chunks: list[RetrievedChunk]
    model: str
    latency_ms: float | None = None
    # Per-stage retrieval wall time (ms): dense (embed+Chroma), bm25, fuse+dedup+slice
    retrieval_timings_ms: dict[str, float] | None = None


# ---------------------------------------------------------------------------
# Metric result models
# ---------------------------------------------------------------------------

class MetricScore(BaseModel):
    """Result of a single metric evaluation on a single sample."""

    metric_name: str
    score: float  # 0.0 - 1.0
    threshold: float
    passed: bool
    reason: str | None = None


class EvalResult(BaseModel):
    """Full evaluation result for a single sample."""

    sample_id: str
    query: str
    generated_answer: str
    retrieval_context: list[str]
    expected_answer: str | None = None
    scores: list[MetricScore]
    latency_ms: float | None = None

    @property
    def passed_all(self) -> bool:
        return all(s.passed for s in self.scores)


class EvalReport(BaseModel):
    """Aggregated evaluation report across all samples."""

    dataset_name: str
    results: list[EvalResult]
    config: dict[str, Any] = Field(default_factory=dict)

    @property
    def pass_rate(self) -> float:
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.passed_all) / len(self.results)

    @property
    def mean_scores(self) -> dict[str, float]:
        """Average score per metric across all samples."""
        from collections import defaultdict

        totals: dict[str, list[float]] = defaultdict(list)
        for result in self.results:
            for score in result.scores:
                totals[score.metric_name].append(score.score)
        return {name: sum(vals) / len(vals) for name, vals in totals.items()}
