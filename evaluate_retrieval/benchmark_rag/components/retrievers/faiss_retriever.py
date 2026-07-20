"""
FAISS-based dense vector retriever.

Supports L2 distance and inner-product (cosine with normalized vectors).
The index is built in-memory from a list of EmbeddedChunks.
For large corpora, call save_index / load_index to persist between runs.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from benchmark_rag.components.base import BaseRetriever, EmbeddedChunk, RetrievedChunk


class FaissRetriever(BaseRetriever):
    """
    Parameters
    ----------
    metric:
        "cosine" (IndexFlatIP on L2-normalised vectors) or "l2" (IndexFlatL2).
    """

    def __init__(self, metric: str = "cosine"):
        if metric not in ("cosine", "l2"):
            raise ValueError(f"metric must be 'cosine' or 'l2', got '{metric}'")
        self.metric = metric
        self._index = None
        self._chunks: list[EmbeddedChunk] = []

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------

    def build_index(self, chunks: list[EmbeddedChunk]) -> None:
        import faiss

        self._chunks = chunks
        emb = np.array([c.embedding for c in chunks], dtype="float32")

        if self.metric == "cosine":
            # L2-normalise so inner product == cosine similarity
            faiss.normalize_L2(emb)
            self._index = faiss.IndexFlatIP(emb.shape[1])
        else:
            self._index = faiss.IndexFlatL2(emb.shape[1])

        self._index.add(emb)

    def add_chunks(self, chunks: list[EmbeddedChunk]) -> None:
        """
        Append new EmbeddedChunks to an already-loaded index.
        Requires build_index() or load_index() to have been called first.
        """
        if self._index is None:
            raise RuntimeError("No index loaded. Call build_index() or load_index() first.")
        if not chunks:
            return

        import faiss

        emb = np.array([c.embedding for c in chunks], dtype="float32")
        if self.metric == "cosine":
            faiss.normalize_L2(emb)
        self._index.add(emb)
        self._chunks.extend(chunks)

    def save_index(self, path: str | Path) -> None:
        import faiss
        import pickle

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(path.with_suffix(".faiss")))
        with open(path.with_suffix(".chunks.pkl"), "wb") as f:
            pickle.dump(self._chunks, f)

    def load_index(self, path: str | Path) -> None:
        import faiss
        import pickle

        path = Path(path)
        self._index = faiss.read_index(str(path.with_suffix(".faiss")))
        with open(path.with_suffix(".chunks.pkl"), "rb") as f:
            self._chunks = pickle.load(f)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(self, query_embedding: list[float], k: int = 5) -> list[RetrievedChunk]:
        if self._index is None:
            raise RuntimeError("Index not built. Call build_index() or load_index() first.")

        import faiss

        q = np.array([query_embedding], dtype="float32")
        if self.metric == "cosine":
            faiss.normalize_L2(q)

        scores, ids = self._index.search(q, k)

        results: list[RetrievedChunk] = []
        for score, idx in zip(scores[0], ids[0]):
            if idx == -1:
                continue
            src = self._chunks[idx]
            results.append(
                RetrievedChunk(
                    text=src.text,
                    doc_id=src.doc_id,
                    chunk_idx=src.chunk_idx,
                    metadata=src.metadata,
                    embedding=src.embedding,
                    score=float(score),
                )
            )
        return results


if __name__ == "__main__":
    import numpy as np
    from benchmark_rag.components.base import EmbeddedChunk

    # Fake 8-dimensional embeddings for 5 chunks
    rng = np.random.default_rng(42)
    dim = 8

    # Two "legal procedure" vectors, two "contract law" vectors, one random
    legal_proc = rng.normal([1, 0, 0, 0, 0, 0, 0, 0], 0.1, (2, dim)).astype("float32")
    contract   = rng.normal([0, 1, 0, 0, 0, 0, 0, 0], 0.1, (2, dim)).astype("float32")
    noise      = rng.normal(0, 1, (1, dim)).astype("float32")
    raw = np.vstack([legal_proc, contract, noise])

    # L2-normalise so cosine retrieval is meaningful
    raw /= np.linalg.norm(raw, axis=1, keepdims=True)

    labels = [
        "Charter s.8 — warrantless search",
        "Charter s.24(2) — exclusion of evidence",
        "Promissory estoppel — no pre-existing relationship",
        "Contract formation — offer and acceptance",
        "Random noise chunk",
    ]

    chunks = [
        EmbeddedChunk(
            text=labels[i], doc_id=f"doc_{i}", chunk_idx=0,
            metadata={}, embedding=raw[i].tolist()
        )
        for i in range(len(labels))
    ]

    for metric in ("cosine", "l2"):
        print(f"\n--- FaissRetriever(metric='{metric}') ---")
        retriever = FaissRetriever(metric=metric)
        retriever.build_index(chunks)

        # Query vector close to the "legal procedure" cluster
        query = rng.normal([1, 0, 0, 0, 0, 0, 0, 0], 0.05, (dim,)).astype("float32")
        query /= np.linalg.norm(query)

        results = retriever.retrieve(query.tolist(), k=3)
        print(f"  Query ~ 'legal procedure cluster' → top {len(results)} results:")
        for r in results:
            print(f"    score={r.score:.4f}  {r.text}")
