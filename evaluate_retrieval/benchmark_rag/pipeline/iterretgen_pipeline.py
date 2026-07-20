"""
IterRetGen pipeline: Iterative Retrieval-then-Generate.

Algorithm (repeated max_iterations times):
  1. Embed the current query (starts as the original query).
  2. Retrieve top-k chunks.
  3. Generate an intermediate answer from the retrieved chunks.
  4. Augment the query: current_query = original_query + "\\n" + intermediate_answer.
  5. Repeat from step 1 with the augmented query.

On the final iteration, step 3-4 are replaced by the regular final-answer generation.
Intermediate generation always uses the original query so the answers stay on-topic.

Usage
-----
    from benchmark_rag.config.schemas import ExperimentConfig
    from benchmark_rag.pipeline.iterretgen_pipeline import IterRetGenPipeline

    cfg = ExperimentConfig.from_yaml("configs/experiments/my_iterretgen_exp.yaml")
    pipeline = IterRetGenPipeline.from_config(cfg)
    result = pipeline.query("What constitutes wrongful dismissal in Ontario?")
"""
from __future__ import annotations

import logging
from pathlib import Path

from benchmark_rag.components.base import BaseEmbedder, BaseGenerator, BaseRetriever, RetrievedChunk
from benchmark_rag.config.schemas import ExperimentConfig
from benchmark_rag.pipeline.rag_pipeline import QueryResult
from benchmark_rag.prompts.iterretgen import (
    FULL_INTERMEDIATE_PROMPT as _FULL_INTERMEDIATE_PROMPT,
    SHORT_INTERMEDIATE_PROMPT as _SHORT_INTERMEDIATE_PROMPT,
)
from benchmark_rag.registry import build_from_component_config

log = logging.getLogger(__name__)


class IterRetGenPipeline:
    """
    Iterative Retrieval-then-Generate pipeline.

    Parameters
    ----------
    embedder:
        Embeds queries (must match the index embedder).
    retriever:
        Pre-loaded retriever (index already built/loaded).
    intermediate_generator:
        Generator used to produce intermediate answers that augment the query.
    generator:
        Generator for the final answer. If None, the last retrieved chunks are
        returned without a generated answer.
    k:
        Number of chunks retrieved per iteration.
    max_iterations:
        Number of retrieval rounds. Default 3.
    """

    def __init__(
        self,
        embedder: BaseEmbedder,
        retriever: BaseRetriever,
        intermediate_generator: BaseGenerator,
        generator: BaseGenerator | None = None,
        k: int = 5,
        max_iterations: int = 3,
    ):
        self.embedder = embedder
        self.retriever = retriever
        self.intermediate_generator = intermediate_generator
        self.generator = generator
        self.k = k
        self.max_iterations = max_iterations

    @classmethod
    def from_config(cls, cfg: ExperimentConfig) -> "IterRetGenPipeline":
        from benchmark_rag.components.generators.gemini import GeminiGenerator
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
        retriever.load_index(Path(cfg.indexing.output_dir) / "index")

        itercfg = cfg.iterretgen
        max_iterations = itercfg.max_iterations if itercfg else 3
        short_answer = itercfg.short_answer if itercfg else True
        intermediate_model = itercfg.intermediate_model_name if itercfg else "gemini-2.5-flash"

        intermediate_max_tokens = itercfg.intermediate_max_output_tokens if itercfg else 200
        intermediate_generator = GeminiGenerator(
            model_name=intermediate_model,
            system_prompt=_SHORT_INTERMEDIATE_PROMPT if short_answer else _FULL_INTERMEDIATE_PROMPT,
            max_output_tokens=intermediate_max_tokens,
        )

        generator = None
        if cfg.generator is not None:
            generator = build_from_component_config(cfg.generator.to_build_dict())

        k = cfg.evaluation.k_values[0] if cfg.evaluation else 5
        return cls(
            embedder=embedder,
            retriever=retriever,
            intermediate_generator=intermediate_generator,
            generator=generator,
            k=k,
            max_iterations=max_iterations,
        )

    def query(self, query_text: str, k: int | None = None) -> QueryResult:
        """
        Run the IterRetGen loop.

        Each iteration retrieves on the current augmented query, generates an
        intermediate answer, and appends it to the original query for the next
        retrieval. The final retrieval's chunks are passed to the main generator.
        """
        k = k or self.k
        current_query = query_text
        chunks: list[RetrievedChunk] = []

        for iteration in range(self.max_iterations):
            query_emb = self.embedder.embed([current_query])[0]
            chunks = self.retriever.retrieve(query_emb, k=k)

            log.info(
                "IterRetGen iteration %d/%d: retrieved %d chunks (query_len=%d)",
                iteration + 1,
                self.max_iterations,
                len(chunks),
                len(current_query),
            )

            # On the last iteration, skip intermediate generation — the final
            # answer generator (below) handles the last set of chunks.
            if iteration < self.max_iterations - 1:
                intermediate_answer = self.intermediate_generator.generate(query_text, chunks)
                current_query = f"{query_text}\n{intermediate_answer}"
                log.debug("Intermediate answer: %s", intermediate_answer[:200])

        answer = None
        if self.generator is not None:
            answer = self.generator.generate(query_text, chunks)

        return QueryResult(
            query=query_text,
            retrieved_chunks=chunks,
            answer=answer,
            metadata={
                "iterretgen_iterations": self.max_iterations,
                "final_augmented_query_len": len(current_query),
            },
        )

    def batch_query(self, queries: list[str], k: int | None = None) -> list[QueryResult]:
        return [self.query(q, k=k) for q in queries]

    def log_usage_summary(self) -> None:
        self.intermediate_generator.log_usage_summary()
        if self.generator is not None:
            self.generator.log_usage_summary()
