"""Fixed-size character chunker with optional overlap."""
from __future__ import annotations
from benchmark_rag.components.base import BaseChunker, Chunk, Document


class NaiveChunker(BaseChunker):
    """
    Splits documents into fixed-size character windows with overlap.
    Fast and deterministic — good baseline.
    """

    def __init__(self, max_chunk_chars: int = 512, overlap_chars: int = 64):
        self.max_chunk_chars = max_chunk_chars
        self.overlap_chars = overlap_chars

    def chunk(self, document: Document) -> list[Chunk]:
        text = document.text
        chunks: list[Chunk] = []
        idx = 0
        chunk_idx = 0
        step = self.max_chunk_chars - self.overlap_chars

        while idx < len(text):
            window = text[idx : idx + self.max_chunk_chars]
            chunks.append(
                Chunk(
                    text=window,
                    doc_id=document.doc_id,
                    chunk_idx=chunk_idx,
                    metadata=dict(document.metadata),
                )
            )
            idx += step
            chunk_idx += 1

        return chunks


if __name__ == "__main__":
    from benchmark_rag.components.base import Document

    sample = Document(
        doc_id="2021 SCC 5",
        text=(
            "The issue in this appeal is whether the doctrine of promissory estoppel applies "
            "where no pre-existing legal relationship exists between the parties. "
            "The Court of Appeal held that it does not. We agree and dismiss the appeal. "
            "Promissory estoppel requires a clear and unequivocal promise, reliance by the "
            "promisee, and detriment suffered as a result of that reliance. "
            "The doctrine operates as a shield, not a sword — it can be used defensively "
            "to prevent a party from resiling from a promise, but cannot found an independent cause of action."
        ),
        metadata={"court": "SCC"},
    )

    for max_chars, overlap in [(128, 16), (256, 32), (512, 64)]:
        chunker = NaiveChunker(max_chunk_chars=max_chars, overlap_chars=overlap)
        chunks = chunker.chunk(sample)
        print(f"\n--- max_chunk_chars={max_chars}, overlap={overlap} → {len(chunks)} chunks ---")
        for c in chunks:
            print(f"  [{c.chunk_idx}] {len(c.text):4d} chars | {c.text[:80]!r}")
