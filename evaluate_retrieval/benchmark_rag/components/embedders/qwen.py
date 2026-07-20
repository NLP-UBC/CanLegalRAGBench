"""Qwen3 embedding model via HuggingFace sentence-transformers."""
from __future__ import annotations

import logging

import torch
from benchmark_rag.components.base import BaseEmbedder

log = logging.getLogger(__name__)


class QwenEmbedder(BaseEmbedder):
    """
    Qwen3-Embedding-8B (or any Qwen3 embedding variant) via sentence-transformers.

    Parameters
    ----------
    model_name:
        HuggingFace model ID, default "Qwen/Qwen3-Embedding-8B".
    device:
        Torch device string, e.g. "cuda:0".
    batch_size:
        Texts per forward pass.
    prompt_name:
        Qwen3 supports task-specific prompt prefixes.
        Pass "query" when embedding queries, None for documents.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Embedding-8B",
        device: str = "cuda:0",
        batch_size: int = 16,
        prompt_name: str | None = None,
    ):
        super().__init__()
        self.model_name = model_name
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.prompt_name = prompt_name
        self._model = None
        self._dim: int | None = None

    def _load(self):
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(self.model_name, device=str(self.device))
        self._dim = self._model.get_sentence_embedding_dimension()

    @property
    def embedding_dim(self) -> int:
        self._load()
        return self._dim  # type: ignore[return-value]

    def _embed(self, texts: list[str]) -> list[list[float]]:
        self._load()
        kwargs: dict = dict(
            sentences=texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        if self.prompt_name is not None:
            kwargs["prompt_name"] = self.prompt_name

        return self._model.encode(**kwargs).tolist()  # type: ignore[union-attr]


if __name__ == "__main__":
    import numpy as np

    texts = [
        "The accused was found guilty of fraud over $5,000.",
        "The appellant submits the trial judge erred in admitting hearsay evidence.",
        "Promissory estoppel requires a clear and unequivocal promise.",
        "The Crown must prove each element of the offence beyond a reasonable doubt.",
        "The quick brown fox jumps over the lazy dog.",  # off-topic — should score low
    ]

    print(f"Loading {QwenEmbedder.__name__} (Qwen/Qwen3-Embedding-0.6B) ...")
    embedder = QwenEmbedder(model_name="Qwen/Qwen3-Embedding-0.6B", device="cpu", batch_size=8)
    embeddings = embedder.embed(texts)

    print(f"Embedding dim : {embedder.embedding_dim}")
    print(f"Vectors shape : {len(embeddings)} x {len(embeddings[0])}")
    print(f"Call count    : {embedder.call_count}")

    emb = np.array(embeddings)
    # Cosine similarity (vectors are L2-normalised by QwenEmbedder)
    sim = emb @ emb.T
    print("\nCosine similarity matrix (legal sentences should cluster together):")
    header = "      " + "  ".join(f"[{i}]" for i in range(len(texts)))
    print(header)
    for i, row in enumerate(sim):
        scores = "  ".join(f"{v:.2f}" for v in row)
        print(f"  [{i}]  {scores}  | {texts[i][:50]}")
