"""
Hybrid RAG pipeline: combines dense (FAISS) and sparse (BM25) retrieval,
then reranks the merged candidate pool with a cross-encoder (e.g. Kanon2Reranker).

Flow with reranker (preferred):
    Query
     ├─ Qwen embedding → FAISS top-N candidates ─┐
     └─ BM25 tokenise  → BM25  top-N candidates ─┴─ union/dedup
                                                      │
                                               Kanon2Reranker
                                               (reads query + each doc text)
                                                      │
                                                   top-k results

Flow without reranker (fallback):
    Fuses the two ranked lists with Reciprocal Rank Fusion (RRF) instead.

Usage
-----
    python scripts/run_benchmark.py --config configs/experiments/hybrid_bm25_qwen_kanon2rerank_recursive_4096.yaml
"""
from __future__ import annotations

from pathlib import Path

from benchmark_rag.components.base import BaseEmbedder, BaseGenerator, BaseReranker
from benchmark_rag.components.retrievers.hybrid_retriever import HybridRetriever
from benchmark_rag.config.schemas import ExperimentConfig
from benchmark_rag.logging import get_logger
from benchmark_rag.pipeline.rag_pipeline import QueryResult
from benchmark_rag.registry import build_from_component_config


class HybridRAGPipeline:
    """
    RAG pipeline that fuses dense and sparse retrieval via a cross-encoder reranker.

    Parameters
    ----------
    embedder:
        Encodes the query for the FAISS dense leg.
    retriever:
        HybridRetriever with both FAISS and BM25 indexes already loaded.
    reranker:
        Cross-encoder that scores the merged candidate pool.  When present,
        retrieve_candidates() is called to get the pool and the reranker
        produces the final ranking.  When None, RRF fusion is used instead.
    generator:
        Optional LLM for answer generation.
    k:
        Final number of results to return after reranking (or RRF).
    n_candidates:
        Candidates to fetch from each sub-retriever before reranking.
        Defaults to ``k * 4`` when not set — gives the reranker enough
        material to work with without sending thousands of texts to the API.
    """

    def __init__(
        self,
        embedder: BaseEmbedder,
        retriever: HybridRetriever,
        reranker: BaseReranker | None = None,
        generator: BaseGenerator | None = None,
        k: int = 5,
        n_candidates: int | None = None,
    ):
        self.embedder = embedder
        self.retriever = retriever
        self.reranker = reranker
        self.generator = generator
        self.k = k
        self.n_candidates = n_candidates
        self.log = get_logger(__name__)

    @classmethod
    def from_config(cls, cfg: ExperimentConfig) -> "HybridRAGPipeline":
        """
        Build a HybridRAGPipeline from an ExperimentConfig.

        Expects the hybrid index (both .faiss/.chunks.pkl and .bm25.pkl) to
        already exist at cfg.indexing.output_dir.  Run run_indexing.py first.
        """
        # Build query-time embedder — switch task type for API embedders that
        # distinguish document vs. query encoding (Gemini, Kanon2 embedder).
        embedder_cfg = cfg.embedder.to_build_dict()
        if "task_type" in embedder_cfg:
            embedder_cfg["task_type"] = "RETRIEVAL_QUERY"
        if "task" in embedder_cfg:
            embedder_cfg["task"] = "retrieval/query"
        embedder: BaseEmbedder = build_from_component_config(embedder_cfg)

        retriever: HybridRetriever = build_from_component_config(cfg.retriever.to_build_dict())
        index_path = Path(cfg.indexing.output_dir) / "index"
        retriever.load_index(index_path)

        reranker: BaseReranker | None = None
        if cfg.reranker is not None:
            reranker = build_from_component_config(cfg.reranker.to_build_dict())

        generator: BaseGenerator | None = None
        if cfg.generator is not None:
            generator = build_from_component_config(cfg.generator.to_build_dict())

        k = max(cfg.evaluation.k_values) if cfg.evaluation else 5
        return cls(
            embedder=embedder,
            retriever=retriever,
            reranker=reranker,
            generator=generator,
            k=k,
        )

    def query(self, query_text: str, k: int | None = None) -> QueryResult:
        """Retrieve and (optionally) rerank candidates, then (optionally) generate."""
        k = k or self.k
        query_emb = self.embedder.embed([query_text])[0]

        if self.reranker is not None:
            # Wide retrieval → cross-encoder reranking
            n = self.n_candidates or k * 4
            candidates = self.retriever.retrieve_candidates(
                query_text, query_emb, n_faiss=n, n_bm25=n
            )
            self.log.debug(
                "HybridRAGPipeline: %d candidates (faiss+bm25 union) → reranker",
                len(candidates),
            )
            chunks = self.reranker.rerank(query_text, candidates)[:k]
        else:
            # RRF fallback
            chunks = self.retriever.retrieve_hybrid(query_text, query_emb, k=k)

        answer = None
        if self.generator is not None:
            answer = self.generator.generate(query_text, chunks)

        return QueryResult(query=query_text, retrieved_chunks=chunks, answer=answer)

    def batch_query(self, queries: list[str], k: int | None = None) -> list[QueryResult]:
        """Query multiple questions, embedding them in one batched call."""
        k = k or self.k
        query_embs = self.embedder.embed(queries)
        results = []
        for query_text, emb in zip(queries, query_embs):
            if self.reranker is not None:
                n = self.n_candidates or k * 4
                candidates = self.retriever.retrieve_candidates(
                    query_text, emb, n_faiss=n, n_bm25=n
                )
                chunks = self.reranker.rerank(query_text, candidates)[:k]
            else:
                chunks = self.retriever.retrieve_hybrid(query_text, emb, k=k)

            answer = None
            if self.generator is not None:
                answer = self.generator.generate(query_text, chunks)
            results.append(QueryResult(query=query_text, retrieved_chunks=chunks, answer=answer))
        return results

    def log_usage_summary(self) -> None:
        """Delegate usage summaries to any components that track API cost."""
        if hasattr(self.reranker, "log_usage_summary"):
            self.reranker.log_usage_summary()
        if self.generator is not None and hasattr(self.generator, "log_usage_summary"):
            self.generator.log_usage_summary()
