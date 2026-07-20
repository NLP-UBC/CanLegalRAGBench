"""EmbeddingGemma-300M embedding model via HuggingFace sentence-transformers."""
from __future__ import annotations

import logging

import torch
from benchmark_rag.components.base import BaseEmbedder

log = logging.getLogger(__name__)


class GemmaEmbedder(BaseEmbedder):
    """
    EmbeddingGemma-300M via sentence-transformers.

    Uses task-specific encoding methods:
      - ``encode_document(texts)`` for indexing (prepends ``title: none | text:``)
      - ``encode_query(texts)`` for retrieval (prepends ``task: search result | query:``)

    The ``task_type`` parameter controls which method is used.  The pipeline
    switching code sets ``task_type = "RETRIEVAL_QUERY"`` at query time;
    any other value (including the default) uses document encoding.

    Parameters
    ----------
    model_name:
        HuggingFace model ID.
    device:
        Torch device string, e.g. "cuda:0".
    batch_size:
        Texts per forward pass.
    task_type:
        ``"RETRIEVAL_QUERY"`` or ``"query"`` → ``encode_query()``.
        Anything else (default ``"RETRIEVAL_DOCUMENT"``) → ``encode_document()``.
    """

    def __init__(
        self,
        model_name: str = "google/embeddinggemma-300m",
        device: str = "cuda:0",
        batch_size: int = 64,
        task_type: str = "RETRIEVAL_DOCUMENT",
    ):
        super().__init__()
        self.model_name = model_name
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.task_type = task_type
        self._model = None
        self._dim: int | None = None

    def _load(self):
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer

        log.info("Loading %s on %s ...", self.model_name, self.device)
        self._model = SentenceTransformer(self.model_name, device=str(self.device))
        self._dim = self._model.get_sentence_embedding_dimension()

        test_emb = self._model.encode(["smoke test"], normalize_embeddings=True)
        assert test_emb.shape == (1, self._dim), (
            f"Unexpected embedding shape: {test_emb.shape}, expected (1, {self._dim})"
        )
        log.info("GemmaEmbedder loaded: dim=%d, task_type=%s", self._dim, self.task_type)

    @property
    def embedding_dim(self) -> int:
        self._load()
        return self._dim  # type: ignore[return-value]

    @property
    def _is_query(self) -> bool:
        return self.task_type.lower() in ("retrieval_query", "query")

    def _embed(self, texts: list[str]) -> list[list[float]]:
        self._load()
        kwargs: dict = dict(
            batch_size=self.batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        if self._is_query:
            return self._model.encode_query(texts, **kwargs).tolist()
        return self._model.encode_document(texts, **kwargs).tolist()


if __name__ == "__main__":
    import numpy as np

    texts = [
        "The accused was found guilty of fraud over $5,000.",
        "The appellant submits the trial judge erred in admitting hearsay evidence.",
        "Promissory estoppel requires a clear and unequivocal promise.",
        "The Crown must prove each element of the offence beyond a reasonable doubt.",
        "The quick brown fox jumps over the lazy dog.",
    ]

    print(f"Loading GemmaEmbedder (document mode) ...")
    doc_embedder = GemmaEmbedder(device="cpu", task_type="RETRIEVAL_DOCUMENT")
    doc_embs = doc_embedder.embed(texts)
    print(f"Embedding dim : {doc_embedder.embedding_dim}")
    print(f"Vectors shape : {len(doc_embs)} x {len(doc_embs[0])}")

    print(f"\nLoading GemmaEmbedder (query mode) ...")
    query_embedder = GemmaEmbedder(device="cpu", task_type="RETRIEVAL_QUERY")
    query = "What must the Crown prove in a criminal case?"
    query_emb = query_embedder.embed([query])

    emb_matrix = np.array(doc_embs)
    q = np.array(query_emb[0])
    sims = emb_matrix @ q
    print(f"\nQuery: {query}")
    print("Similarities:")
    for i, (s, t) in enumerate(zip(sims, texts)):
        print(f"  [{i}] {s:.4f}  {t[:60]}")
