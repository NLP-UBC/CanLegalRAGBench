"""
Abstract base classes for all swappable RAG components.

Every component follows the same contract:
  - Instantiated with a validated Pydantic config (its own sub-schema).
  - One primary method that takes / returns well-typed data.
  - No side-effects beyond what is declared (no global state, no hardcoded paths).

Adding a new component: subclass the right ABC and register it in registry.py.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


class BudgetExceededError(Exception):
    """Raised when a component's cost cap is exceeded."""
    pass


# ---------------------------------------------------------------------------
# Shared data containers
# ---------------------------------------------------------------------------


@dataclass
class Chunk:
    """A single text chunk produced by a Chunker."""

    text: str
    doc_id: str
    chunk_idx: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EmbeddedChunk(Chunk):
    """A Chunk enriched with its embedding vector."""

    embedding: list[float] = field(default_factory=list)


@dataclass
class RetrievedChunk(EmbeddedChunk):
    """An EmbeddedChunk returned by a Retriever, with a relevance score."""

    score: float = 0.0


@dataclass
class Document:
    """Raw source document before chunking."""

    doc_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Splitter  (sentence / paragraph / token boundary detection)
# ---------------------------------------------------------------------------


class BaseSplitter(ABC):
    """
    Splits a raw string into atomic units (sentences, paragraphs, …).
    Used internally by Chunkers — not called directly by the pipeline.
    """

    @abstractmethod
    def split(self, text: str) -> list[str]:
        """Return a list of string segments."""


# ---------------------------------------------------------------------------
# Chunker  (assembles Splitter output into Chunks)
# ---------------------------------------------------------------------------


class BaseChunker(ABC):
    """
    Turns a Document into a list of Chunks, optionally using a Splitter.
    Implementations decide the chunking strategy (naive, semantic, …).
    """

    @abstractmethod
    def chunk(self, document: Document) -> list[Chunk]:
        """Return an ordered list of Chunks for a single Document."""

    def batch_chunk(self, documents: list[Document]) -> list[list[Chunk]]:
        """Chunk a list of documents. Override for batched optimisation."""
        return [self.chunk(doc) for doc in documents]


# ---------------------------------------------------------------------------
# Embedder
# ---------------------------------------------------------------------------


class BaseEmbedder(ABC):
    """
    Encodes a list of strings into dense vectors.

    Concrete classes wrap HuggingFace, VoyageAI, Gemini, etc.
    All return plain Python float lists so downstream code stays framework-agnostic.
    """

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Dimensionality of the output vectors."""

    @abstractmethod
    def _embed(self, texts: list[str]) -> list[list[float]]:
        """Backend-specific embedding logic."""

    def embed(self, texts: list[str], **kwargs) -> list[list[float]]:
        """Public entry point — adds call counting and delegates to _embed."""
        self._call_count += 1
        return self._embed(texts, **kwargs)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

    def __init__(self):
        self._call_count: int = 0

    @property
    def call_count(self) -> int:
        return self._call_count

    def reset_call_count(self):
        self._call_count = 0


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------


class BaseRetriever(ABC):
    """
    Searches an index and returns the top-k most relevant EmbeddedChunks.

    Concrete implementations: FAISS (dense), BM25 (sparse), hybrid.
    The index is built once via `build_index`; queries are then served by `retrieve`.
    """

    @abstractmethod
    def build_index(self, chunks: list[EmbeddedChunk]) -> None:
        """Populate the internal index from a list of embedded chunks."""

    @abstractmethod
    def retrieve(self, query_embedding: list[float], k: int = 5) -> list[RetrievedChunk]:
        """Return the k most relevant chunks for a query embedding."""

    def batch_retrieve(
        self, query_embeddings: list[list[float]], k: int = 5
    ) -> list[list[RetrievedChunk]]:
        """Retrieve for multiple queries. Override for batched optimisation."""
        return [self.retrieve(qe, k) for qe in query_embeddings]


# ---------------------------------------------------------------------------
# Reranker
# ---------------------------------------------------------------------------


class BaseReranker(ABC):
    """
    Re-scores a candidate set of RetrievedChunks given the original query text.

    Typical usage: pass the top-k from a fast retriever, get back a re-ranked list.
    """

    @abstractmethod
    def rerank(self, query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """Return chunks sorted by reranked relevance (highest first)."""


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


class BaseGenerator(ABC):
    """
    Generates a natural-language answer from a query and retrieved context.

    Concrete implementations wrap Anthropic, OpenAI, local LLMs, etc.
    """

    @abstractmethod
    def generate(self, query: str, context_chunks: list[RetrievedChunk]) -> str:
        """Return a generated answer string."""
