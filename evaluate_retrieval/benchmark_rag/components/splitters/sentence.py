"""NLTK sentence splitter."""
from __future__ import annotations
from benchmark_rag.components.base import BaseSplitter


class SentenceSplitter(BaseSplitter):
    """Splits text into sentences using NLTK's Punkt tokenizer."""

    def __init__(self, nltk_data_dir: str | None = None):
        import nltk

        if nltk_data_dir:
            nltk.download("punkt", download_dir=nltk_data_dir, quiet=True)
            nltk.download("punkt_tab", download_dir=nltk_data_dir, quiet=True)
        else:
            nltk.download("punkt", quiet=True)
            nltk.download("punkt_tab", quiet=True)

        self._tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")

    def split(self, text: str) -> list[str]:
        return self._tokenizer.tokenize(text)
