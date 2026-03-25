"""LLM response generation via LM Studio (OpenAI-compatible chat completions)."""

from __future__ import annotations

import time
from dataclasses import dataclass

import mlflow
from mlflow.entities.span import SpanType
from openai import OpenAI

from src.data.schemas import RAGResult, RetrievedChunk

DEFAULT_SYSTEM_PROMPT = """\
You are a technical assistant. Answer questions using ONLY the provided context.
Do not add information from your own knowledge — if the context does not contain
enough information, say what is missing rather than guessing.

Context:
{context}
"""


@dataclass
class Generator:
    """Generate answers using a local model served by LM Studio.

    Usage:
        client = openai_client(LMStudioSettings.from_config(cfg))
        gen = Generator(client=client, model="...")
        result = gen.generate("What is X?", retrieved_chunks)
    """

    client: OpenAI
    model: str
    temperature: float = 0.0
    max_tokens: int = 1024
    system_prompt: str = DEFAULT_SYSTEM_PROMPT

    @mlflow.trace(span_type=SpanType.LLM)
    def generate(self, query: str, retrieved_chunks: list[RetrievedChunk]) -> RAGResult:
        """Generate a response grounded in retrieved context.

        Args:
            query: The user's question.
            retrieved_chunks: Chunks from the retriever.

        Returns:
            RAGResult with the answer and metadata.
        """
        context = self._format_context(retrieved_chunks)
        system = self.system_prompt.format(context=context, query=query)

        start = time.perf_counter()
        create_kw: dict = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": query},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        completion = self.client.chat.completions.create(**create_kw)
        latency_ms = (time.perf_counter() - start) * 1000

        msg = completion.choices[0].message
        answer = msg.content or ""

        return RAGResult(
            query=query,
            answer=answer,
            retrieved_chunks=retrieved_chunks,
            model=self.model,
            latency_ms=latency_ms,
        )

    def _format_context(self, chunks: list[RetrievedChunk]) -> str:
        """Format retrieved chunks into a context string for the prompt."""
        parts = []
        for i, rc in enumerate(chunks, 1):
            source = rc.chunk.metadata.source_path
            parts.append(f"[{i}] (source: {source})\n{rc.chunk.text}")
        return "\n\n---\n\n".join(parts)
