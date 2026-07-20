"""
Isaacus Kanon2 embedding model via the isaacus Python SDK.

Kanon2 is a legal-domain embedding model served via the Isaacus API.
API reference: https://docs.isaacus.com
"""
from __future__ import annotations

import logging

from benchmark_rag.components.base import BaseEmbedder

log = logging.getLogger(__name__)


class Kanon2Embedder(BaseEmbedder):
    """
    Embeds text using the Isaacus Kanon2 legal embedding model.

    Parameters
    ----------
    model_name:
        Kanon2 model variant, e.g. "kanon-2-embedder".
    api_key:
        Isaacus API key.  Falls back to ISAACUS_API_KEY env var if None.
    batch_size:
        Texts per API request.
    task:
        Isaacus task type.  Use "retrieval/document" for corpus chunks,
        "retrieval/query" for query embeddings.
    max_cost_usd:
        Maximum estimated spend in USD before raising RuntimeError.
    """

    _EMBEDDING_DIM = 1024  # update if the model version changes
    _COST_PER_1M_TOKENS = 0.35  # USD per million tokens (Isaacus pricing)

    def __init__(
        self,
        model_name: str = "kanon-2-embedder",
        api_key: str | None = None,
        batch_size: int = 64,
        task: str = "retrieval/document",
        max_cost_usd: float | None = None,
    ):
        super().__init__()
        self.model_name = model_name
        self.batch_size = batch_size
        self._api_key = api_key
        self.task = task
        self.max_cost_usd = max_cost_usd
        self._client = None
        self._total_prompt_tokens: int = 0
        self._total_est_cost_usd: float = 0.0
        self._tokens_reported: bool = False

    def _load(self):
        if self._client is not None:
            return
        import os
        from isaacus import Isaacus

        key = self._api_key or os.environ.get("ISAACUS_API_KEY")
        if not key:
            raise EnvironmentError(
                "No Isaacus API key found. Set ISAACUS_API_KEY or pass api_key= to Kanon2Embedder."
            )
        self._client = Isaacus(api_key=key)

    @property
    def embedding_dim(self) -> int:
        return self._EMBEDDING_DIM

    def _track_and_log(self, n_texts: int, call_prompt_tokens: int | None, call_est_tokens: int) -> None:
        if call_prompt_tokens is not None:
            self._total_prompt_tokens += call_prompt_tokens
            self._tokens_reported = True
            call_cost = call_prompt_tokens / 1_000_000 * self._COST_PER_1M_TOKENS
            self._total_est_cost_usd += call_cost
            log.info(
                "Kanon2Embedder model=%s | embed call: texts=%d prompt_tokens=%d cost=$%.6f"
                " | running total prompt_tokens=%d est_cost=$%.4f",
                self.model_name, n_texts, call_prompt_tokens, call_cost,
                self._total_prompt_tokens, self._total_est_cost_usd,
            )
        else:
            call_cost = call_est_tokens / 1_000_000 * self._COST_PER_1M_TOKENS
            self._total_est_cost_usd += call_cost
            log.info(
                "Kanon2Embedder model=%s | embed call: texts=%d prompt_tokens=N/A"
                " est_cost=$%.6f (estimate = chars/4) | running total est_cost=$%.4f",
                self.model_name, n_texts, call_cost, self._total_est_cost_usd,
            )

    def log_usage_summary(self) -> None:
        if self._tokens_reported:
            log.info(
                "Kanon2Embedder usage summary | model=%s | total_prompt_tokens=%d"
                " total_cost=$%.4f (limit=$%s)",
                self.model_name, self._total_prompt_tokens, self._total_est_cost_usd,
                f"{self.max_cost_usd:.2f}" if self.max_cost_usd is not None else "none",
            )
        else:
            log.info(
                "Kanon2Embedder usage summary | model=%s | prompt_tokens=N/A"
                " total_est_cost=$%.4f (estimate = chars/4, limit=$%s)",
                self.model_name, self._total_est_cost_usd,
                f"{self.max_cost_usd:.2f}" if self.max_cost_usd is not None else "none",
            )

    def _embed(self, texts: list[str]) -> list[list[float]]:
        self._load()
        all_embeddings: list[list[float]] = []
        n_batches = (len(texts) + self.batch_size - 1) // self.batch_size
        call_prompt_tokens: int | None = None
        call_est_tokens: int = 0

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            batch_est_tokens = sum(len(t) for t in batch) // 4
            batch_est_cost = batch_est_tokens / 1_000_000 * self._COST_PER_1M_TOKENS
            if self.max_cost_usd is not None:
                if self._total_est_cost_usd + batch_est_cost > self.max_cost_usd:
                    from benchmark_rag.components.base import BudgetExceededError
                    raise BudgetExceededError(
                        f"Kanon2Embedder cost limit of ${self.max_cost_usd:.2f} would be exceeded "
                        f"(accumulated: ${self._total_est_cost_usd:.4f}, "
                        f"next batch est: ${batch_est_cost:.4f}). "
                        f"Stopping before batch {i // self.batch_size + 1}/{n_batches}."
                    )
            response = self._client.embeddings.create(  # type: ignore[union-attr]
                model=self.model_name,
                texts=batch,
                task=self.task,
            )
            batch_embeddings = [e.embedding for e in response.embeddings]
            all_embeddings.extend(batch_embeddings)
            usage = response.usage
            pt = getattr(usage, "prompt_tokens", None) or getattr(usage, "total_tokens", None)
            if pt is not None:
                call_prompt_tokens = (call_prompt_tokens or 0) + int(pt)
            call_est_tokens += batch_est_tokens
            log.debug(
                "Kanon2Embedder batch %d/%d: texts=%d prompt_tokens=%s",
                i // self.batch_size + 1, n_batches, len(batch),
                pt if pt is not None else "N/A",
            )

        self._track_and_log(len(texts), call_prompt_tokens, call_est_tokens)
        return all_embeddings


if __name__ == "__main__":
    import os
    import numpy as np

    if not os.environ.get("ISAACUS_API_KEY"):
        print("ISAACUS_API_KEY not set — skipping Kanon2Embedder test.")
    else:
        texts = [
            "The accused was found guilty of fraud over $5,000.",
            "The appellant submits the trial judge erred in admitting hearsay evidence.",
            "Promissory estoppel requires a clear and unequivocal promise.",
            "The Crown must prove each element beyond a reasonable doubt.",
            "The quick brown fox jumps over the lazy dog.",
        ]

        print("Loading Kanon2Embedder ...")
        embedder = Kanon2Embedder()
        embeddings = embedder.embed(texts)

        print(f"Embedding dim : {embedder.embedding_dim}")
        print(f"Vectors shape : {len(embeddings)} x {len(embeddings[0])}")
        print(f"Call count    : {embedder.call_count}")
        embedder.log_usage_summary()

        emb = np.array(embeddings)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        emb_norm = emb / norms
        sim = emb_norm @ emb_norm.T
        print("\nCosine similarity matrix:")
        for i, row in enumerate(sim):
            scores = "  ".join(f"{v:.2f}" for v in row)
            print(f"  [{i}]  {scores}  | {texts[i][:50]}")
