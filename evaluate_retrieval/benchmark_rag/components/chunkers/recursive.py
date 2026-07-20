"""
Recursive character text splitter.

Tries to split on progressively smaller separators (paragraph → sentence →
word → character) so chunk boundaries fall on natural text units where possible.
Produces Chunks of at most `max_chunk_chars` with configurable overlap.
"""
from __future__ import annotations

from benchmark_rag.components.base import BaseChunker, Chunk, Document

_DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


class RecursiveChunker(BaseChunker):
    """
    Mirrors LangChain's RecursiveCharacterTextSplitter logic but operates on
    our Document / Chunk dataclasses with no LangChain dependency.

    Parameters
    ----------
    max_chunk_chars:
        Hard upper bound on chunk length in characters.
    overlap_chars:
        Characters of overlap carried from the previous chunk.
    separators:
        Ordered list of separator strings to try.  Falls back to the next
        separator when the current one produces pieces that are still too large.
    """

    def __init__(
        self,
        max_chunk_chars: int = 1024,
        overlap_chars: int = 128,
        separators: list[str] | None = None,
    ):
        self.max_chunk_chars = max_chunk_chars
        self.overlap_chars = overlap_chars
        self.separators = separators if separators is not None else _DEFAULT_SEPARATORS

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk(self, document: Document) -> list[Chunk]:
        raw_chunks = self._split(document.text, self.separators)
        merged = self._merge(raw_chunks)
        return [
            Chunk(
                text=text,
                doc_id=document.doc_id,
                chunk_idx=i,
                metadata=dict(document.metadata),
            )
            for i, text in enumerate(merged)
        ]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _split(self, text: str, separators: list[str]) -> list[str]:
        """Recursively split text using the first separator that applies."""
        if not text:
            return []

        # Find the first separator that actually exists in the text
        sep = ""
        remaining_seps: list[str] = []
        for i, s in enumerate(separators):
            if s == "" or s in text:
                sep = s
                remaining_seps = separators[i + 1 :]
                break

        if sep == "":
            # Character-level fallback: just return as-is (merge will window it)
            return [text]

        splits = text.split(sep)
        # Re-attach separator to preserve structure (except for whitespace seps)
        if sep.strip():
            splits = [s + sep for s in splits[:-1]] + [splits[-1]]

        good: list[str] = []
        for piece in splits:
            if not piece:
                continue
            if len(piece) <= self.max_chunk_chars:
                good.append(piece)
            else:
                # Piece is still too large — recurse with narrower separators
                good.extend(self._split(piece, remaining_seps or [""]))

        return good

    def _merge(self, splits: list[str]) -> list[str]:
        """Greedily merge small splits into windows ≤ max_chunk_chars with overlap."""
        chunks: list[str] = []
        current: list[str] = []
        current_len = 0

        for piece in splits:
            piece_len = len(piece)

            if current_len + piece_len > self.max_chunk_chars and current:
                chunks.append("".join(current))
                # Keep overlap: drop from front until we're within overlap budget
                while current and current_len > self.overlap_chars:
                    removed = current.pop(0)
                    current_len -= len(removed)

            current.append(piece)
            current_len += piece_len

        if current:
            chunks.append("".join(current))

        return chunks


if __name__ == "__main__":
    from benchmark_rag.components.base import Document

    sample = Document(
        doc_id="2022 ONCA 100",
        text=(
            "R. v. Smith, 2022 ONCA 100\n\n"
            "The appellant was convicted of fraud over $5,000 following a trial by judge alone. "
            "He now appeals his conviction on the ground that the trial judge erred in admitting "
            "certain documentary evidence obtained during a warrantless search of his business premises.\n\n"
            "The Crown concedes the search was warrantless but argues the documents fall within "
            "the consent exception. The trial judge agreed, finding that the appellant's office "
            "manager had apparent authority to consent on his behalf.\n\n"
            "We disagree. The office manager had no actual or apparent authority to waive the "
            "appellant's Charter rights. The documents should have been excluded under s. 24(2). "
            "Without them, the conviction cannot stand. The appeal is allowed and an acquittal entered.\n\n"
            "The test under s. 24(2) requires the court to consider all the circumstances, including "
            "the seriousness of the Charter-infringing conduct, the impact of admitting the evidence "
            "on the repute of the justice system, and the society's interest in adjudication on the merits. "
            "In this case, the warrantless entry into commercial premises was a serious breach. "
            "The evidence is excluded."
        ),
        metadata={"court": "ONCA", "year": 2022},
    )

    for max_chars in (256, 512, 1024):
        chunker = RecursiveChunker(max_chunk_chars=max_chars, overlap_chars=64)
        chunks = chunker.chunk(sample)
        print(f"\n--- max_chunk_chars={max_chars} → {len(chunks)} chunks ---")
        for c in chunks:
            print(f"  [{c.chunk_idx}] {len(c.text):4d} chars | {c.text[:80].replace(chr(10), ' ')!r}")
