"""
Semantic chunker — merges sentences until cosine similarity drops.

Mirrors the TopicChunker from cad_rag but:
  - Takes a BaseSplitter and BaseEmbedder via dependency injection
    (no hardcoded imports of specific models).
  - Uses the unified Chunk dataclass.
  - No breakpoint() or debug artefacts.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from benchmark_rag.components.base import BaseChunker, BaseEmbedder, BaseSplitter, Chunk, Document


class SemanticChunker(BaseChunker):
    """
    Groups sentences into chunks based on embedding similarity.

    Starts a new chunk whenever the cosine similarity between adjacent
    sentences falls below `similarity_threshold` OR the chunk exceeds
    `max_chunk_chars_hard` characters.
    """

    def __init__(
        self,
        splitter: BaseSplitter,
        embedder: BaseEmbedder,
        similarity_threshold: float = 0.6,
        max_chunk_chars_hard: int = 2048,
    ):
        self.splitter = splitter
        self.embedder = embedder
        self.similarity_threshold = similarity_threshold
        self.max_chunk_chars_hard = max_chunk_chars_hard

    def chunk(self, document: Document) -> list[Chunk]:
        sentences = self.splitter.split(document.text)
        if not sentences:
            return []

        embeddings = torch.tensor(self.embedder.embed(sentences))  # [N, D]

        # Cosine similarity between each sentence and its successor
        shifted = torch.roll(embeddings, -1, dims=0)
        sims = F.cosine_similarity(embeddings, shifted, dim=1)  # [N]

        chunks: list[Chunk] = []
        current: list[str] = []
        current_len = 0
        chunk_idx = 0

        for i, sentence in enumerate(sentences):
            current.append(sentence)
            current_len += len(sentence)

            is_last = i == len(sentences) - 1
            over_size = current_len >= self.max_chunk_chars_hard
            topic_break = not is_last and sims[i].item() < self.similarity_threshold

            if is_last or over_size or topic_break:
                chunks.append(
                    Chunk(
                        text=" ".join(current),
                        doc_id=document.doc_id,
                        chunk_idx=chunk_idx,
                        metadata=dict(document.metadata),
                    )
                )
                current = []
                current_len = 0
                chunk_idx += 1

        return chunks
