"""Document chunking strategies.

Phase 1: Recursive character splitting with document-type-aware separators.
Future phases will add semantic chunking, AST-based code splitting, etc.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from src.data.schemas import Chunk, DocumentMeta, DocumentType


# Default separators per document type
DEFAULT_SEPARATORS: dict[DocumentType, list[str]] = {
    DocumentType.MARKDOWN: ["\n## ", "\n### ", "\n#### ", "\n\n", "\n", " "],
    DocumentType.CODE: ["\nclass ", "\ndef ", "\nasync def ", "\n\n", "\n", " "],
    DocumentType.MERMAID: ["\n\n", "\n"],
    DocumentType.TEXT: ["\n\n", "\n", ". ", " "],
}


@dataclass
class RecursiveChunker:
    """Split documents into chunks using recursive character splitting.

    Tries to split on the most semantically meaningful separator first
    (e.g., heading boundaries for markdown, function boundaries for code),
    falling back to smaller separators as needed to stay within chunk_size.
    """

    chunk_size: int = 512
    chunk_overlap: int = 64
    separators: list[str] | None = None
    _doc_type_overrides: dict[DocumentType, dict] = field(default_factory=dict)

    def with_overrides(self, overrides: dict[DocumentType, dict]) -> RecursiveChunker:
        """Create a new chunker with document-type-specific settings."""
        new = RecursiveChunker(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
        )
        new._doc_type_overrides = overrides
        return new

    def chunk_document(self, content: str, metadata: DocumentMeta) -> list[Chunk]:
        """Split a document into chunks.

        Args:
            content: Full document text.
            metadata: Document metadata (used to select separators).

        Returns:
            List of Chunk objects with IDs and metadata.
        """
        # Apply document-type-specific overrides
        chunk_size = self.chunk_size
        chunk_overlap = self.chunk_overlap
        separators = self.separators

        if metadata.doc_type in self._doc_type_overrides:
            overrides = self._doc_type_overrides[metadata.doc_type]
            chunk_size = overrides.get("chunk_size", chunk_size)
            chunk_overlap = overrides.get("chunk_overlap", chunk_overlap)
            if "separators" in overrides:
                separators = overrides["separators"]

        if separators is None:
            separators = DEFAULT_SEPARATORS.get(metadata.doc_type, DEFAULT_SEPARATORS[DocumentType.TEXT])

        raw_chunks = self._recursive_split(content, separators, chunk_size)
        chunks = self._apply_overlap(raw_chunks, chunk_overlap, content)

        return [
            Chunk(
                chunk_id=f"{metadata.source_path}::{uuid.uuid4().hex[:8]}",
                text=text,
                metadata=metadata,
                start_char=start,
                end_char=start + len(text),
            )
            for text, start in chunks
        ]

    def _recursive_split(
        self,
        text: str,
        separators: list[str],
        chunk_size: int,
    ) -> list[str]:
        """Recursively split text, trying larger separators first."""
        if len(text) <= chunk_size:
            return [text] if text.strip() else []

        if not separators:
            # Last resort: hard split at chunk_size
            return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

        sep = separators[0]
        remaining_separators = separators[1:]
        parts = text.split(sep)

        result = []
        current = ""

        for i, part in enumerate(parts):
            # Re-attach the separator (except for the first part)
            candidate = (sep + part) if i > 0 else part

            if len(current) + len(candidate) <= chunk_size:
                current += candidate
            else:
                if current.strip():
                    result.append(current)
                # If this single part exceeds chunk_size, split it further
                if len(candidate) > chunk_size:
                    result.extend(self._recursive_split(candidate, remaining_separators, chunk_size))
                    current = ""
                else:
                    current = candidate

        if current.strip():
            result.append(current)

        return result

    def _apply_overlap(
        self,
        chunks: list[str],
        overlap: int,
        original: str,
    ) -> list[tuple[str, int]]:
        """Add overlap between consecutive chunks. Returns (text, start_offset) pairs."""
        if not chunks or overlap <= 0:
            offset = 0
            result = []
            for chunk in chunks:
                idx = original.find(chunk, offset)
                if idx == -1:
                    idx = offset
                result.append((chunk, idx))
                offset = idx + len(chunk)
            return result

        result = []
        offset = 0
        for i, chunk in enumerate(chunks):
            idx = original.find(chunk, offset)
            if idx == -1:
                idx = offset

            if i > 0 and idx >= overlap:
                # Prepend overlap from previous chunk's tail
                overlap_text = original[idx - overlap : idx]
                chunk = overlap_text + chunk

            result.append((chunk, max(0, idx - overlap) if i > 0 else idx))
            offset = idx + len(chunks[i])  # Use original chunk length for offset

        return result
