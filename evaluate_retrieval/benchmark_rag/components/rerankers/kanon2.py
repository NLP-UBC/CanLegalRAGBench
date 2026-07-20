"""
Isaacus Kanon2 cross-encoder reranker via the isaacus Python SDK.

API reference: https://docs.isaacus.com

The reranker reads the raw query text and each candidate document together,
producing a true relevance score rather than a positional proxy.  This makes
it significantly stronger than rank-fusion heuristics for legal retrieval.

Pricing: $0.35 per million tokens ($0.00000035 per token).
Token usage and running cost are logged after every call and summarised at the
end of a run via log_usage_summary().
"""
from __future__ import annotations

import logging
import os

from benchmark_rag.components.base import BaseReranker, RetrievedChunk

log = logging.getLogger(__name__)


class Kanon2Reranker(BaseReranker):
    """
    Reranks a candidate pool using the Isaacus Kanon2 legal reranker model.

    The reranker sends (query, candidate_text) pairs to the Isaacus API and
    receives a relevance score for each pair.  Results are returned sorted by
    score descending.

    Parameters
    ----------
    model_name:
        Kanon2 reranker variant (default "kanon-2-reranker").
    api_key:
        Isaacus API key. Falls back to ISAACUS_API_KEY env var if None.
    batch_size:
        Maximum number of texts per API call.  The Isaacus API may impose
        per-request limits; keep this conservative to avoid errors.
    max_cost_usd:
        Hard spend cap in USD.  Raises RuntimeError before a batch that
        would push the estimated cost over this limit.
    """

    _COST_PER_1M_TOKENS = 0.35  # USD per million tokens (Isaacus pricing)

    def __init__(
        self,
        model_name: str = "kanon-2-reranker",
        api_key: str | None = None,
        batch_size: int = 100,
        max_cost_usd: float | None = None,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self._api_key = api_key
        self.max_cost_usd = max_cost_usd
        self._client = None
        self._total_prompt_tokens: int = 0
        self._total_est_cost_usd: float = 0.0
        self._tokens_reported: bool = False

    def _load(self) -> None:
        if self._client is not None:
            return
        from isaacus import Isaacus

        key = self._api_key or os.environ.get("ISAACUS_API_KEY")
        if not key:
            raise EnvironmentError(
                "No Isaacus API key found. "
                "Set ISAACUS_API_KEY or pass api_key= to Kanon2Reranker."
            )
        self._client = Isaacus(api_key=key)

    def _track_and_log(
        self,
        n_texts: int,
        call_prompt_tokens: int | None,
        call_est_tokens: int,
    ) -> None:
        if call_prompt_tokens is not None:
            self._total_prompt_tokens += call_prompt_tokens
            self._tokens_reported = True
            call_cost = call_prompt_tokens / 1_000_000 * self._COST_PER_1M_TOKENS
            self._total_est_cost_usd += call_cost
            log.info(
                "Kanon2Reranker model=%s | rerank call: texts=%d prompt_tokens=%d cost=$%.6f"
                " | running total prompt_tokens=%d est_cost=$%.4f",
                self.model_name, n_texts, call_prompt_tokens, call_cost,
                self._total_prompt_tokens, self._total_est_cost_usd,
            )
        else:
            call_cost = call_est_tokens / 1_000_000 * self._COST_PER_1M_TOKENS
            self._total_est_cost_usd += call_cost
            log.info(
                "Kanon2Reranker model=%s | rerank call: texts=%d prompt_tokens=N/A"
                " est_cost=$%.6f (estimate = chars/4) | running total est_cost=$%.4f",
                self.model_name, n_texts, call_cost, self._total_est_cost_usd,
            )

    def log_usage_summary(self) -> None:
        if self._tokens_reported:
            log.info(
                "Kanon2Reranker usage summary | model=%s | total_prompt_tokens=%d"
                " total_cost=$%.4f (limit=$%s)",
                self.model_name, self._total_prompt_tokens, self._total_est_cost_usd,
                f"{self.max_cost_usd:.2f}" if self.max_cost_usd is not None else "none",
            )
        else:
            log.info(
                "Kanon2Reranker usage summary | model=%s | prompt_tokens=N/A"
                " total_est_cost=$%.4f (estimate = chars/4, limit=$%s)",
                self.model_name, self._total_est_cost_usd,
                f"{self.max_cost_usd:.2f}" if self.max_cost_usd is not None else "none",
            )

    def rerank(self, query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """
        Score every chunk against the query and return the list sorted by
        relevance score descending.

        Chunks are processed in batches of ``batch_size``.  Each batch
        makes one API call.  Token usage and running cost are logged after
        every call; call ``log_usage_summary()`` at the end of a run for
        the aggregate.
        """
        if not chunks:
            return chunks

        self._load()
        texts = [c.text for c in chunks]

        scored: list[tuple[int, float]] = []  # (original_index, reranker_score)
        call_prompt_tokens: int | None = None
        call_est_tokens: int = 0

        for start in range(0, len(texts), self.batch_size):
            batch_texts = texts[start : start + self.batch_size]
            # Estimate tokens as (query chars + all candidate chars) / 4
            batch_est_tokens = (len(query) + sum(len(t) for t in batch_texts)) // 4
            batch_est_cost = batch_est_tokens / 1_000_000 * self._COST_PER_1M_TOKENS

            if self.max_cost_usd is not None:
                if self._total_est_cost_usd + batch_est_cost > self.max_cost_usd:
                    from benchmark_rag.components.base import BudgetExceededError
                    raise BudgetExceededError(
                        f"Kanon2Reranker cost limit of ${self.max_cost_usd:.2f} would be exceeded "
                        f"(accumulated: ${self._total_est_cost_usd:.4f}, "
                        f"next batch est: ${batch_est_cost:.4f}). Stopping."
                    )

            response = self._client.rerankings.create(  # type: ignore[union-attr]
                model=self.model_name,
                query=query,
                texts=batch_texts,
            )
            for result in response.results:
                scored.append((start + result.index, result.score))

            usage = getattr(response, "usage", None)
            if usage is not None:
                pt = getattr(usage, "prompt_tokens", None) or getattr(usage, "total_tokens", None)
                if pt is not None:
                    call_prompt_tokens = (call_prompt_tokens or 0) + int(pt)
            call_est_tokens += batch_est_tokens

            log.debug(
                "Kanon2Reranker batch [%d:%d]: scored %d texts",
                start, start + len(batch_texts), len(batch_texts),
            )

        self._track_and_log(len(texts), call_prompt_tokens, call_est_tokens)

        scored.sort(key=lambda x: x[1], reverse=True)
        reranked = [
            RetrievedChunk(
                text=chunks[orig_idx].text,
                doc_id=chunks[orig_idx].doc_id,
                chunk_idx=chunks[orig_idx].chunk_idx,
                metadata=chunks[orig_idx].metadata,
                embedding=chunks[orig_idx].embedding,
                score=score,
            )
            for orig_idx, score in scored
        ]

        log.info(
            "Kanon2Reranker: scored %d candidates | top doc_id=%s (score=%.4f)",
            len(chunks),
            reranked[0].doc_id if reranked else "N/A",
            reranked[0].score if reranked else 0.0,
        )
        return reranked
