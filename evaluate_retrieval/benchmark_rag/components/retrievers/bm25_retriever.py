"""
BM25 sparse retriever built on rank-bm25 (BM25L).

Install the optional dependency before use:
    pip install rank-bm25

Because BM25 scores documents against raw query text — not an embedding vector —
this retriever exposes retrieve_text(query_text, k) as its primary entry point.
The BaseRetriever.retrieve(query_embedding, k) method is implemented but raises
a RuntimeError directing callers to retrieve_text().

BM25L vs BM25Okapi
------------------
BM25Okapi penalises long documents more harshly than necessary: once a term's
normalised TF saturates near k1, adding more occurrences doesn't help, but being
in a long document still hurts.  BM25L fixes this by introducing a lower-bound
delta on the normalised TF before saturation, so long documents aren't
over-penalised.  delta=0.5 is the value recommended in the original paper
(Lv & Zhai, 2011).

Index level
-----------
index_level="chunk" (default): one BM25 entry per chunk — current behaviour.
index_level="document": one BM25 entry per full document.  TF and length
normalisation operate on the original document text, avoiding the distortion
caused by chunking.  At retrieval time, results are document-level (one per
doc_id); callers that need chunk-level output (e.g. HybridRetriever) expand
using the stored doc_id → chunks mapping.
"""
from __future__ import annotations

import logging
import pickle
import re
from collections import defaultdict
from pathlib import Path

from benchmark_rag.components.base import BaseRetriever, Document, EmbeddedChunk, RetrievedChunk

log = logging.getLogger(__name__)


def _tokenize(text: str) -> list[str]:
    """Lowercase and split on non-alphanumeric characters."""
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25Retriever(BaseRetriever):
    """
    Sparse BM25 retriever (BM25L or BM25Okapi).

    Parameters
    ----------
    variant:
        "L" (default) — BM25L with delta lower-bound on normalised TF.
        "okapi" — standard BM25Okapi.
    k1:
        Term-frequency saturation parameter (default 1.5).
    b:
        Length-normalisation parameter (default 0.75).
    delta:
        BM25L lower-bound on normalised TF (default 0.5).
        Only used when variant="L".
    index_level:
        "chunk" (default) — one BM25 entry per chunk.
        "document" — one BM25 entry per full document.  Requires passing
        documents to build_index().
    """

    def __init__(
        self,
        variant: str = "L",
        k1: float = 1.5,
        b: float = 0.75,
        delta: float = 0.5,
        index_level: str = "chunk",
    ):
        if index_level not in ("chunk", "document"):
            raise ValueError(f"index_level must be 'chunk' or 'document', got '{index_level}'")
        if variant not in ("L", "okapi"):
            raise ValueError(f"variant must be 'L' or 'okapi', got '{variant}'")
        self.variant = variant
        self.k1 = k1
        self.b = b
        self.delta = delta
        self.index_level = index_level
        self._bm25 = None
        self._chunks: list[EmbeddedChunk] = []
        self._doc_order: list[str] = []
        self._doc_chunk_lookup: dict[str, list[EmbeddedChunk]] = {}

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------

    def _make_bm25(self, tokenized_corpus: list[list[str]]):
        if self.variant == "okapi":
            from rank_bm25 import BM25Okapi
            return BM25Okapi(tokenized_corpus, k1=self.k1, b=self.b)
        else:
            from rank_bm25 import BM25L
            return BM25L(tokenized_corpus, k1=self.k1, b=self.b, delta=self.delta)

    def build_index(
        self,
        chunks: list[EmbeddedChunk],
        documents: list[Document] | None = None,
    ) -> None:
        self._chunks = chunks

        if self.index_level == "document":
            if documents is None:
                raise ValueError(
                    "index_level='document' requires documents to be passed to build_index(). "
                    "Pass the original Document list from the indexing pipeline."
                )
            self._build_document_level(documents, chunks)
        else:
            tokenized_corpus = [_tokenize(c.text) for c in chunks]
            self._bm25 = self._make_bm25(tokenized_corpus)

    def _build_document_level(
        self,
        documents: list[Document],
        chunks: list[EmbeddedChunk],
    ) -> None:
        self._doc_order = [d.doc_id for d in documents]

        lookup: dict[str, list[EmbeddedChunk]] = defaultdict(list)
        for c in chunks:
            lookup[c.doc_id].append(c)
        for chunk_list in lookup.values():
            chunk_list.sort(key=lambda c: c.chunk_idx)
        self._doc_chunk_lookup = dict(lookup)

        tokenized_corpus = [_tokenize(d.text) for d in documents]
        self._bm25 = self._make_bm25(tokenized_corpus)

        log.info(
            "BM25 document-level index built: %d documents, %d chunks",
            len(documents), len(chunks),
        )

    def _bm25_suffix(self) -> str:
        """Filename suffix for the BM25 pickle, keyed by index_level."""
        if self.index_level == "document":
            return ".bm25_doc.pkl"
        return ".bm25.pkl"

    def save_index(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        out = path.parent / (path.stem + self._bm25_suffix())
        with open(out, "wb") as f:
            pickle.dump({
                "bm25": self._bm25,
                "chunks": self._chunks,
                "index_level": self.index_level,
                "doc_order": self._doc_order,
                "doc_chunk_lookup": self._doc_chunk_lookup,
            }, f)

    def load_index(self, path: str | Path) -> None:
        path = Path(path)
        pkl = path.parent / (path.stem + self._bm25_suffix())
        if not pkl.exists() and self.index_level == "chunk":
            pkl = path.with_suffix(".bm25.pkl")
        with open(pkl, "rb") as f:
            data = pickle.load(f)
        self._bm25 = data["bm25"]
        self._chunks = data["chunks"]
        self.index_level = data.get("index_level", "chunk")
        self._doc_order = data.get("doc_order", [])
        self._doc_chunk_lookup = data.get("doc_chunk_lookup", {})

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve_text(self, query_text: str, k: int = 5) -> list[RetrievedChunk]:
        """
        Primary entry point for BM25 — takes raw query text.

        In chunk mode: returns one RetrievedChunk per matching chunk.
        In document mode: returns one RetrievedChunk per matching document
        (the first chunk is used as a representative).
        """
        if self._bm25 is None:
            raise RuntimeError("Index not built. Call build_index() or load_index() first.")

        scores = self._bm25.get_scores(_tokenize(query_text))
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

        if self.index_level == "document":
            return self._retrieve_doc_level(scores, top_indices)
        return self._retrieve_chunk_level(scores, top_indices)

    def _retrieve_chunk_level(self, scores, top_indices: list[int]) -> list[RetrievedChunk]:
        return [
            RetrievedChunk(
                text=self._chunks[i].text,
                doc_id=self._chunks[i].doc_id,
                chunk_idx=self._chunks[i].chunk_idx,
                metadata=self._chunks[i].metadata,
                embedding=self._chunks[i].embedding,
                score=float(scores[i]),
            )
            for i in top_indices
        ]

    def _retrieve_doc_level(self, scores, top_indices: list[int]) -> list[RetrievedChunk]:
        results = []
        for idx in top_indices:
            doc_id = self._doc_order[idx]
            doc_chunks = self._doc_chunk_lookup.get(doc_id, [])
            if not doc_chunks:
                continue
            first = doc_chunks[0]
            results.append(RetrievedChunk(
                text=first.text,
                doc_id=doc_id,
                chunk_idx=first.chunk_idx,
                metadata=first.metadata,
                embedding=first.embedding,
                score=float(scores[idx]),
            ))
        return results

    def retrieve(self, query_embedding: list[float], k: int = 5) -> list[RetrievedChunk]:
        raise RuntimeError(
            "BM25Retriever requires query text, not an embedding. "
            "Use retrieve_text(query_text, k) instead."
        )
