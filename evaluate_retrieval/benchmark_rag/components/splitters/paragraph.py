"""Paragraph splitter — splits on blank lines."""
from __future__ import annotations
import re
from benchmark_rag.components.base import BaseSplitter


class ParagraphSplitter(BaseSplitter):
    """Splits text on one or more blank lines."""

    def split(self, text: str) -> list[str]:
        paragraphs = re.split(r"\n\s*\n", text)
        return [p.strip() for p in paragraphs if p.strip()]
