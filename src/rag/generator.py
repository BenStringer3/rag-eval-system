"""LLM response generation via Ollama.

Takes a query and retrieved context, formats a prompt, and generates
a grounded response using a local LLM.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import ollama

from src.data.schemas import RAGResult, RetrievedChunk

DEFAULT_SYSTEM_PROMPT = """\
You are a helpful assistant that answers questions based on the provided context.
Always ground your answers in the retrieved context. If the context doesn't contain
enough information to answer, say so explicitly.

Context:
{context}
"""


@dataclass
class Generator:
    """Generate answers using a local Ollama LLM.

    Usage:
        gen = Generator()
        result = gen.generate("What is X?", retrieved_chunks)
    """

    model: str = "qwen3:14b"
    temperature: float = 0.1
    max_tokens: int = 1024
    system_prompt: str = DEFAULT_SYSTEM_PROMPT

    def generate(self, query: str, retrieved_chunks: list[RetrievedChunk]) -> RAGResult:
        """Generate a response grounded in retrieved context.

        Args:
            query: The user's question.
            retrieved_chunks: Chunks from the retriever.

        Returns:
            RAGResult with the answer and metadata.
        """
        context = self._format_context(retrieved_chunks)
        system = self.system_prompt.format(context=context)

        start = time.perf_counter()
        response = ollama.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": query},
            ],
            options={
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        )
        latency_ms = (time.perf_counter() - start) * 1000

        answer = response["message"]["content"]

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
