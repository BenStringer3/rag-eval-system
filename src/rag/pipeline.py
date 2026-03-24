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
from src.rag.bm25_index import Bm25Index
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
    hybrid_index_path: Path | None = None
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

        retrieval_cfg = cfg.get("retrieval", {})
        hybrid_cfg = retrieval_cfg.get("hybrid") or {}
        hybrid_enabled = hybrid_cfg.get("enabled", True)
        candidate_k = int(hybrid_cfg.get("candidate_k", 20))
        rrf_k = int(hybrid_cfg.get("rrf_k", 60))
        fusion = hybrid_cfg.get("fusion", "rrf")
        if fusion != "rrf":
            raise ValueError(f"Unsupported hybrid fusion: {fusion!r} (only 'rrf' is implemented)")
        hybrid_index_path = Path(
            hybrid_cfg.get("index_path", "./data/embeddings/bm25_index.pkl")
        )

        bm25: Bm25Index | None = None
        hybrid_path_for_pipeline: Path | None = None
        if hybrid_enabled:
            hybrid_path_for_pipeline = hybrid_index_path
            if hybrid_index_path.exists():
                bm25 = Bm25Index.load(hybrid_index_path)
            elif store.count() > 0:
                # Chroma populated but no BM25 file (e.g. upgrade): build from store.
                rebuild = Bm25Index()
                rebuild.build(store.get_all_stored_chunks())
                rebuild.save(hybrid_index_path)
                bm25 = rebuild

        retriever = Retriever(
            store=store,
            top_k=retrieval_cfg["top_k"],
            score_threshold=retrieval_cfg.get("score_threshold"),
            hybrid_enabled=hybrid_enabled,
            bm25_index=bm25,
            candidate_k=candidate_k,
            rrf_k=rrf_k,
        )

        gen_cfg = cfg["generation"]
        generator = Generator(
            client=client,
            model=gen_cfg["model"],
            temperature=gen_cfg["temperature"],
            max_tokens=gen_cfg["max_tokens"],
            system_prompt=gen_cfg.get("system_prompt", Generator.system_prompt),
        )

        return cls(
            chunker=chunker,
            embedder=embedder,
            store=store,
            retriever=retriever,
            generator=generator,
            hybrid_index_path=hybrid_path_for_pipeline,
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
        if self.hybrid_index_path is not None:
            print("  Building BM25 index...")
            idx = Bm25Index()
            idx.build(all_chunks)
            idx.save(self.hybrid_index_path)
            self.retriever.bm25_index = idx
        self._ingested = True

        print(f"  Done. Store contains {self.store.count()} chunks.")
        return len(all_chunks)

    def reset_storage(self) -> None:
        """Drop Chroma collection and remove BM25 artifact (same as a full re-ingest prep)."""
        self.store.reset()
        if self.hybrid_index_path is not None and self.hybrid_index_path.exists():
            self.hybrid_index_path.unlink()
        self.retriever.bm25_index = None

    def query(self, question: str) -> RAGResult:
        """Run the full RAG pipeline for a question.

        Args:
            question: Natural language question.

        Returns:
            RAGResult with the generated answer and retrieved context.
        """
        outcome = self.retriever.retrieve_detailed(question)
        result = self.generator.generate(question, outcome.chunks)
        return result.model_copy(update={"retrieval_timings_ms": outcome.timings_ms})

    def query_for_eval(self, question: str) -> tuple[str, list[str]]:
        """Run the pipeline and return (answer, retrieval_context).

        Convenience method for DeepEval test cases which need the
        answer and context as separate values.
        """
        result = self.query(question)
        context = [rc.chunk.text for rc in result.retrieved_chunks]
        return result.answer, context
