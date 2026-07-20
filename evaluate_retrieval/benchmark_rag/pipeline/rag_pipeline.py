"""
RAG query pipeline: query → embedding → retrieval → (optional generation).

Usage
-----
    from benchmark_rag.config.schemas import ExperimentConfig
    from benchmark_rag.pipeline.rag_pipeline import RAGPipeline

    cfg = ExperimentConfig.from_yaml("configs/experiments/qwen_1024.yaml")
    pipeline = RAGPipeline.from_config(cfg)

    result = pipeline.query("What constitutes wrongful dismissal in Ontario?")
    print(result.answer)
    for chunk in result.retrieved_chunks:
        print(chunk.score, chunk.text[:80])
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from benchmark_rag.components.base import BaseEmbedder, BaseGenerator, BaseReranker, BaseRetriever, RetrievedChunk
from benchmark_rag.config.schemas import ExperimentConfig
from benchmark_rag.logging import get_logger
from benchmark_rag.registry import build_from_component_config


@dataclass
class QueryResult:
    query: str
    retrieved_chunks: list[RetrievedChunk]
    answer: str | None = None
    metadata: dict = field(default_factory=dict)


class RAGPipeline:
    """
    Stateless query pipeline.  Components are injected at construction time
    so the same pipeline can answer many queries without reloading models.

    Parameters
    ----------
    embedder:
        Used to embed incoming queries (should match the index embedder).
    retriever:
        Pre-loaded retriever (index must already be built/loaded).
    generator:
        Optional LLM for answer generation.  If None, only retrieval is done.
    reranker:
        Optional cross-encoder reranker.  When set, retrieval widens to
        ``candidate_pool_size`` candidates, then the reranker scores and
        truncates to the final ``k``.
    k:
        Number of chunks to retrieve.
    candidate_pool_size:
        How many candidates to fetch before reranking.  Defaults to ``k * 4``.
        Ignored when no reranker is configured.
    """

    def __init__(
        self,
        embedder: BaseEmbedder,
        retriever: BaseRetriever,
        generator: BaseGenerator | None = None,
        reranker: BaseReranker | None = None,
        k: int = 5,
        candidate_pool_size: int | None = None,
    ):
        self.embedder = embedder
        self.retriever = retriever
        self.generator = generator
        self.reranker = reranker
        self.k = k
        self.candidate_pool_size = candidate_pool_size or (k * 4)
        self.log = get_logger(__name__)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, cfg: ExperimentConfig) -> "RAGPipeline":
        """
        Build a RAGPipeline from an ExperimentConfig.
        Loads the FAISS index from the indexing output directory.
        """
        from benchmark_rag.components.retrievers.faiss_retriever import FaissRetriever

        # Build embedder for query-time — switch task type for API embedders that
        # distinguish document vs. query encoding (Gemini, Kanon2).
        embedder_cfg = cfg.embedder.to_build_dict()
        if "task_type" in embedder_cfg:
            embedder_cfg["task_type"] = "RETRIEVAL_QUERY"
        if "task" in embedder_cfg:
            embedder_cfg["task"] = "retrieval/query"
        embedder: BaseEmbedder = build_from_component_config(embedder_cfg)

        retriever = FaissRetriever(metric=cfg.retriever.model_extra.get("metric", "cosine"))
        index_path = Path(cfg.indexing.output_dir) / "index"
        retriever.load_index(index_path)

        generator = None
        if cfg.generator is not None:
            generator = build_from_component_config(cfg.generator.to_build_dict())

        reranker = None
        if cfg.reranker is not None:
            reranker = build_from_component_config(cfg.reranker.to_build_dict())

        k = max(cfg.evaluation.k_values) if cfg.evaluation else 5
        return cls(embedder=embedder, retriever=retriever, generator=generator, reranker=reranker, k=k)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(self, query_text: str, k: int | None = None) -> QueryResult:
        """Embed query, retrieve top-k chunks, optionally rerank, optionally generate."""
        k = k or self.k

        query_emb = self.embedder.embed([query_text])[0]

        if self.reranker is not None:
            n_candidates = max(k * 4, self.candidate_pool_size)
            candidates = self.retriever.retrieve(query_emb, k=n_candidates)
            self.log.debug("RAGPipeline: %d candidates → reranker", len(candidates))
            chunks = self.reranker.rerank(query_text, candidates)[:k]
        else:
            chunks = self.retriever.retrieve(query_emb, k=k)

        answer = None
        if self.generator is not None:
            answer = self.generator.generate(query_text, chunks)

        return QueryResult(query=query_text, retrieved_chunks=chunks, answer=answer)

    def log_usage_summary(self) -> None:
        if self.reranker is not None and hasattr(self.reranker, "log_usage_summary"):
            self.reranker.log_usage_summary()
        if self.generator is not None and hasattr(self.generator, "log_usage_summary"):
            self.generator.log_usage_summary()

    def batch_query(self, queries: list[str], k: int | None = None) -> list[QueryResult]:
        """Query multiple questions, embedding them in one batched call."""
        k = k or self.k

        query_embs = self.embedder.embed(queries)
        results = []
        for query_text, emb in zip(queries, query_embs):
            chunks = self.retriever.retrieve(emb, k=k)
            answer = None
            if self.generator is not None:
                answer = self.generator.generate(query_text, chunks)
            results.append(QueryResult(query=query_text, retrieved_chunks=chunks, answer=answer))

        return results
