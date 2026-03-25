"""Tests for MLflow retrieval trace payload extraction."""

from __future__ import annotations

from pathlib import Path

import mlflow
from mlflow.genai.utils.trace_utils import extract_retrieval_context_from_trace

from src.data.schemas import Chunk, DocumentMeta, DocumentType, RetrievedChunk
from src.rag.retriever import Retriever


@mlflow.trace
def _run_traced_retrieval(retriever: Retriever, chunks: list[RetrievedChunk]) -> None:
    retriever.trace_retrieval("What is MLflow?", chunks)


def test_trace_retrieval_emits_context_mlflow_can_extract(tmp_path: Path) -> None:
    mlflow.set_tracking_uri((tmp_path / "mlruns").as_uri())

    retriever = Retriever(store=None)  # type: ignore[arg-type]
    chunk = Chunk(
        chunk_id="c1",
        text="MLflow tracks experiments.",
        metadata=DocumentMeta(source_path="docs/mlflow.md", doc_type=DocumentType.MARKDOWN),
    )
    retrieved = RetrievedChunk(chunk=chunk, score=0.9)

    _run_traced_retrieval(retriever, [retrieved])

    trace_id = mlflow.get_last_active_trace_id()
    assert trace_id is not None
    trace = mlflow.get_trace(trace_id)
    assert trace is not None

    contexts = extract_retrieval_context_from_trace(trace)
    assert len(contexts) == 1
    extracted = next(iter(contexts.values()))
    assert extracted == [{"content": "MLflow tracks experiments.", "doc_uri": "docs/mlflow.md"}]
