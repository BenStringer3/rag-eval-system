"""File loaders for different document types.

Handles markdown, code files, and mermaid diagrams. Each loader
extracts text content and attaches appropriate metadata for
downstream chunking and retrieval.
"""

from __future__ import annotations

import re
from pathlib import Path

from src.data.schemas import DocumentMeta, DocumentType


# Map file extensions to document types
EXTENSION_MAP: dict[str, DocumentType] = {
    ".md": DocumentType.MARKDOWN,
    ".markdown": DocumentType.MARKDOWN,
    ".mmd": DocumentType.MERMAID,
    ".mermaid": DocumentType.MERMAID,
    ".py": DocumentType.CODE,
    ".js": DocumentType.CODE,
    ".ts": DocumentType.CODE,
    ".tsx": DocumentType.CODE,
    ".jsx": DocumentType.CODE,
    ".rs": DocumentType.CODE,
    ".go": DocumentType.CODE,
    ".java": DocumentType.CODE,
    ".cpp": DocumentType.CODE,
    ".c": DocumentType.CODE,
    ".h": DocumentType.CODE,
    ".yaml": DocumentType.CODE,
    ".yml": DocumentType.CODE,
    ".toml": DocumentType.CODE,
    ".json": DocumentType.CODE,
    ".txt": DocumentType.TEXT,
}

# Map extensions to language names (for code files)
LANGUAGE_MAP: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".jsx": "javascript",
    ".rs": "rust",
    ".go": "go",
    ".java": "java",
    ".cpp": "cpp",
    ".c": "c",
    ".h": "c",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".json": "json",
}


def detect_doc_type(path: Path) -> DocumentType:
    """Detect document type from file extension."""
    # Check if a .md file contains primarily mermaid content
    if path.suffix in (".md", ".markdown"):
        content = path.read_text(encoding="utf-8", errors="replace")
        if _is_mermaid_dominant(content):
            return DocumentType.MERMAID
    return EXTENSION_MAP.get(path.suffix, DocumentType.TEXT)


def _is_mermaid_dominant(content: str) -> bool:
    """Heuristic: if >50% of the file is inside mermaid code fences."""
    mermaid_blocks = re.findall(r"```mermaid\s*\n(.*?)```", content, re.DOTALL)
    mermaid_chars = sum(len(block) for block in mermaid_blocks)
    total_chars = len(content.strip())
    if total_chars == 0:
        return False
    return mermaid_chars / total_chars > 0.5


def load_document(path: Path) -> tuple[str, DocumentMeta]:
    """Load a single document and return (content, metadata).

    Args:
        path: Path to the document file.

    Returns:
        Tuple of (text content, document metadata).
    """
    content = path.read_text(encoding="utf-8", errors="replace")
    doc_type = detect_doc_type(path)
    title = _extract_title(content, doc_type, path)
    language = LANGUAGE_MAP.get(path.suffix)

    meta = DocumentMeta(
        source_path=str(path),
        doc_type=doc_type,
        title=title,
        language=language,
    )
    return content, meta


def load_corpus(corpus_dir: str | Path) -> list[tuple[str, DocumentMeta]]:
    """Recursively load all supported documents from a directory.

    Args:
        corpus_dir: Root directory to scan.

    Returns:
        List of (content, metadata) tuples.
    """
    corpus_dir = Path(corpus_dir)
    if not corpus_dir.is_dir():
        raise FileNotFoundError(f"Corpus directory not found: {corpus_dir}")

    documents = []
    supported_extensions = set(EXTENSION_MAP.keys())

    for path in sorted(corpus_dir.rglob("*")):
        if path.is_file() and path.suffix in supported_extensions:
            # Skip hidden files and common non-content directories
            parts = path.relative_to(corpus_dir).parts
            if any(p.startswith(".") or p in ("node_modules", "__pycache__", ".git", "logs") for p in parts):
                continue
            try:
                doc = load_document(path)
                documents.append(doc)
            except Exception as e:
                print(f"Warning: Failed to load {path}: {e}")

    return documents


def _extract_title(content: str, doc_type: DocumentType, path: Path) -> str:
    """Best-effort title extraction from document content."""
    if doc_type == DocumentType.MARKDOWN:
        # First H1 heading
        match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
        if match:
            return match.group(1).strip()

    if doc_type == DocumentType.CODE:
        # Module docstring (Python)
        match = re.search(r'^"""(.+?)"""', content, re.DOTALL)
        if match:
            first_line = match.group(1).strip().split("\n")[0]
            return first_line

    # Fallback to filename
    return path.stem
