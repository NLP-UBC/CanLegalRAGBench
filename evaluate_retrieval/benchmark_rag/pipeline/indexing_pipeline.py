"""
Indexing pipeline: Document → Chunks → EmbeddedChunks → FAISS index.

Responsibilities:
  - Load documents from a dataset
  - Chunk with the configured chunker
  - Embed in batches with the configured embedder
  - Build and persist the FAISS index
  - Save chunk metadata to parquet for later inspection

Usage
-----
    from benchmark_rag.config.schemas import ExperimentConfig
    from benchmark_rag.pipeline.indexing_pipeline import IndexingPipeline

    cfg = ExperimentConfig.from_yaml("configs/experiments/qwen_1024.yaml")
    pipeline = IndexingPipeline(cfg)
    pipeline.run(documents)
"""
from __future__ import annotations

import inspect
import pickle
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from benchmark_rag.components.base import BaseChunker, BaseEmbedder, BaseRetriever, Document, EmbeddedChunk
from benchmark_rag.config.schemas import ExperimentConfig
from benchmark_rag.logging import get_logger, log_resource_snapshot
from benchmark_rag.registry import build_from_component_config


class IndexingPipeline:
    """
    Runs the full indexing flow for one experiment configuration.

    Parameters
    ----------
    cfg:
        Validated ExperimentConfig.  All component construction is deferred
        until run() is called so the object is cheap to create.
    """

    def __init__(self, cfg: ExperimentConfig, force_reindex: bool = False):
        self.cfg = cfg
        self._force_reindex = force_reindex
        self.log = get_logger(__name__)
        self._output_dir = Path(cfg.indexing.output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, documents: list[Document]) -> Path:
        """
        Index a list of Documents.

        If an index already exists and force_reindex=False, only documents
        whose doc_id is not already present are chunked, embedded, and appended.
        Returns the path to the saved FAISS index directory.
        """
        cfg = self.cfg
        t0 = time.perf_counter()
        index_path = self._output_dir / "index"
        index_exists = (
            not self._force_reindex
            and (self._output_dir / "index.faiss").exists()
            and (self._output_dir / "index.chunks.pkl").exists()
        )

        chunker: BaseChunker = build_from_component_config(cfg.chunker.to_build_dict())
        embedder: BaseEmbedder = build_from_component_config(cfg.embedder.to_build_dict())
        retriever: BaseRetriever = build_from_component_config(cfg.retriever.to_build_dict())

        if index_exists:
            # --- Incremental path ---
            self.log.info("Existing index found — running incremental update.")
            already_indexed = self._load_indexed_doc_ids()
            new_documents = [d for d in documents if d.doc_id not in already_indexed]
            n_skip = len(documents) - len(new_documents)

            self.log.info(f"Documents in dataset : {len(documents)}")
            self.log.info(f"Already indexed      : {len(already_indexed)} unique doc_ids")
            self.log.info(f"Skipping             : {n_skip} documents (already in index)")
            self.log.info(f"To index             : {len(new_documents)} new documents")

            if not new_documents:
                self.log.info("Nothing to add — index is already up to date.")
                return self._output_dir

            self.log.info(f"Loading existing index from {index_path}...")
            retriever.load_index(index_path)
            self.log.info(
                f"Loaded index with {retriever._index.ntotal} vectors "
                f"across {len(retriever._chunks)} chunks"
            )

            new_chunks = []
            for doc in tqdm(new_documents, desc="Chunking new docs"):
                new_chunks.extend(chunker.chunk(doc))
            self.log.info(
                f"Chunking complete: {len(new_chunks)} new chunks "
                f"from {len(new_documents)} documents "
                f"(avg {len(new_chunks)/len(new_documents):.1f} chunks/doc)"
            )

            embedded_new = self._embed_chunks(new_chunks, embedder)

            chunks_before = len(retriever._chunks)
            retriever.add_chunks(embedded_new)
            retriever.save_index(index_path)
            self.log.info(
                f"Index updated: {chunks_before} → {retriever._index.ntotal} vectors "
                f"(+{retriever._index.ntotal - chunks_before})"
            )
            self.log.info(f"Index saved to {index_path}")

            self._append_metadata(embedded_new)

        else:
            # --- Full build path ---
            if self._force_reindex:
                self.log.info("Force reindex requested — rebuilding from scratch.")
            self.log.info(
                f"Indexing {len(documents)} documents | experiment={cfg.experiment_id}"
            )

            all_chunks = []
            for doc in tqdm(documents, desc="Chunking"):
                all_chunks.extend(chunker.chunk(doc))
            self.log.info(
                f"Chunking complete: {len(all_chunks)} chunks from {len(documents)} documents "
                f"(avg {len(all_chunks)/len(documents):.1f} chunks/doc)"
            )

            embedded_chunks = self._embed_chunks(all_chunks, embedder)

            self.log.info("Building index...")
            if "documents" in inspect.signature(retriever.build_index).parameters:
                retriever.build_index(embedded_chunks, documents=documents)
            else:
                retriever.build_index(embedded_chunks)
            retriever.save_index(index_path)
            self.log.info(
                f"Index built: {retriever._index.ntotal} vectors | saved to {index_path}"
            )

            self._save_metadata(embedded_chunks)

        elapsed = time.perf_counter() - t0
        self.log.info(f"Indexing complete in {elapsed:.1f}s")
        if hasattr(embedder, "log_usage_summary"):
            embedder.log_usage_summary()
        log_resource_snapshot(self.log)

        # Stash so external callers (scripts) can introspect cost trackers.
        self.embedder = embedder

        return self._output_dir

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _embed_chunks(self, chunks, embedder: BaseEmbedder) -> list[EmbeddedChunk]:
        from benchmark_rag.components.base import EmbeddedChunk

        batch_size = self.cfg.indexing.embedding_batch_size
        embedded: list[EmbeddedChunk] = []
        save_every = self.cfg.indexing.save_intermediate_every

        n_batches = (len(chunks) + batch_size - 1) // batch_size
        for i in tqdm(range(0, len(chunks), batch_size), total=n_batches, desc="Embedding", unit="batch"):
            batch = chunks[i : i + batch_size]
            texts = [c.text for c in batch]
            embeddings = embedder.embed(texts)

            for chunk, emb in zip(batch, embeddings):
                embedded.append(
                    EmbeddedChunk(
                        text=chunk.text,
                        doc_id=chunk.doc_id,
                        chunk_idx=chunk.chunk_idx,
                        metadata=chunk.metadata,
                        embedding=emb,
                    )
                )

            if len(embedded) % save_every < batch_size:
                self.log.info(f"Embedded {len(embedded)}/{len(chunks)} chunks")
                log_resource_snapshot(self.log)

        return embedded

    def _load_indexed_doc_ids(self) -> set[str]:
        """Return the set of doc_ids already present in the saved index."""
        meta_path = self._output_dir / "chunks_metadata.parquet"
        if not meta_path.exists():
            return set()
        df = pd.read_parquet(meta_path, columns=["doc_id"])
        return set(df["doc_id"].unique())

    def _save_metadata(self, chunks: list[EmbeddedChunk]) -> None:
        """Save chunk text + metadata (without embeddings) to parquet."""
        rows = [
            {
                "doc_id": c.doc_id,
                "chunk_idx": c.chunk_idx,
                "text": c.text,
                **c.metadata,
            }
            for c in chunks
        ]
        df = pd.DataFrame(rows)
        out = self._output_dir / "chunks_metadata.parquet"
        df.to_parquet(out, index=False)
        self.log.info(f"Chunk metadata saved to {out}")

    def _append_metadata(self, chunks: list[EmbeddedChunk]) -> None:
        """Append new chunk rows to the existing chunks_metadata.parquet."""
        meta_path = self._output_dir / "chunks_metadata.parquet"
        new_rows = [
            {"doc_id": c.doc_id, "chunk_idx": c.chunk_idx, "text": c.text, **c.metadata}
            for c in chunks
        ]
        new_df = pd.DataFrame(new_rows)
        existing_df = pd.read_parquet(meta_path)
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        combined_df.to_parquet(meta_path, index=False)
        self.log.info(f"Appended {len(new_rows)} rows to {meta_path} (total: {len(combined_df)})")
