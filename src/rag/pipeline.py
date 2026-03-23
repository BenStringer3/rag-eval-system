"""End-to-end RAG pipeline.

Wires together chunker, embedder, store, retriever, and generator
into a single callable pipeline. This is the main entry point for
both interactive use and evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from src.data.loaders import load_corpus
from src.data.schemas import RAGResult
from src.inference.lm_studio import LMStudioSettings, openai_client
from src.rag.chunker import RecursiveChunker
from src.rag.embedder import Embedder
from src.rag.generator import Generator
from src.rag.retriever import Retriever
from src.rag.store import VectorStore


@dataclass
class RAGPipeline:
    """Full RAG pipeline: ingest documents, query, and generate answers.

    Usage:
        pipeline = RAGPipeline.from_config("configs/default.yaml")
        pipeline.ingest("data/corpus")
        result = pipeline.query("How does chunking work?")
    """

    chunker: RecursiveChunker
    embedder: Embedder
    store: VectorStore
    retriever: Retriever
    generator: Generator
    _ingested: bool = field(default=False, init=False)

    @classmethod
    def from_config(cls, config_path: str = "configs/default.yaml") -> RAGPipeline:
        """Create a pipeline from a YAML config file."""
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")

        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        lm = LMStudioSettings.from_config(cfg)
        client = openai_client(lm)

        embedder = Embedder(
            client=client,
            model=cfg["embedding"]["model"],
            query_prefix=cfg["embedding"].get("query_prefix", "search_query: "),
            document_prefix=cfg["embedding"].get("document_prefix", "search_document: "),
        )

        chunker = RecursiveChunker(
            chunk_size=cfg["chunking"]["chunk_size"],
            chunk_overlap=cfg["chunking"]["chunk_overlap"],
        )
        # Apply document-type overrides if present
        if "overrides" in cfg["chunking"]:
            from src.data.schemas import DocumentType

            overrides = {}
            for dtype_str, settings in cfg["chunking"]["overrides"].items():
                dtype = DocumentType(dtype_str)
                overrides[dtype] = settings
            chunker = chunker.with_overrides(overrides)

        store = VectorStore(
            embedder=embedder,
            persist_directory=cfg["vector_store"]["persist_directory"],
            collection_name=cfg["vector_store"]["collection_name"],
            distance_metric=cfg["vector_store"]["distance_metric"],
        )

        retriever = Retriever(
            store=store,
            top_k=cfg["retrieval"]["top_k"],
            score_threshold=cfg["retrieval"].get("score_threshold"),
        )

        generator = Generator(
            client=client,
            model=cfg["generation"]["model"],
            temperature=cfg["generation"]["temperature"],
            max_tokens=cfg["generation"]["max_tokens"],
            system_prompt=cfg["generation"].get("system_prompt", Generator.system_prompt),
        )

        return cls(
            chunker=chunker,
            embedder=embedder,
            store=store,
            retriever=retriever,
            generator=generator,
        )

    def ingest(self, corpus_dir: str | Path) -> int:
        """Load, chunk, embed, and store all documents from a directory.

        Args:
            corpus_dir: Path to the corpus directory.

        Returns:
            Number of chunks indexed.
        """
        print(f"Loading documents from {corpus_dir}...")
        documents = load_corpus(corpus_dir)
        print(f"  Found {len(documents)} documents")

        all_chunks = []
        for content, meta in documents:
            chunks = self.chunker.chunk_document(content, meta)
            all_chunks.extend(chunks)

        print(f"  Split into {len(all_chunks)} chunks")
        print("  Embedding and indexing...")
        self.store.add_chunks(all_chunks)
        self._ingested = True

        print(f"  Done. Store contains {self.store.count()} chunks.")
        return len(all_chunks)

    def query(self, question: str) -> RAGResult:
        """Run the full RAG pipeline for a question.

        Args:
            question: Natural language question.

        Returns:
            RAGResult with the generated answer and retrieved context.
        """
        retrieved = self.retriever.retrieve(question)
        result = self.generator.generate(question, retrieved)
        return result

    def query_for_eval(self, question: str) -> tuple[str, list[str]]:
        """Run the pipeline and return (answer, retrieval_context).

        Convenience method for DeepEval test cases which need the
        answer and context as separate values.
        """
        result = self.query(question)
        context = [rc.chunk.text for rc in result.retrieved_chunks]
        return result.answer, context
