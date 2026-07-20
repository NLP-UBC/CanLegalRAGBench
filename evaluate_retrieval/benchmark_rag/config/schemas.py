"""
Pydantic v2 config schemas.

Every section of the YAML maps to one model here, giving:
  - Type-safe access (no dict["key"] typos)
  - Validated defaults
  - Clear documentation of every knob

Loading a config
----------------
    from benchmark_rag.config.schemas import ExperimentConfig
    cfg = ExperimentConfig.from_yaml("configs/experiments/my_exp.yaml")
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Component sub-configs
# Each has a required "type" field (resolved by registry.build) plus
# component-specific fields with sensible defaults.
# ---------------------------------------------------------------------------


class ComponentConfig(BaseModel):
    """Base for any component sub-config. 'type' is the registry key."""

    type: str
    model_config = {"extra": "allow"}  # forward unknown keys to the constructor

    def to_build_dict(self) -> dict[str, Any]:
        """Return a dict ready for registry.build_from_component_config."""
        return self.model_dump()


class SplitterConfig(ComponentConfig):
    type: str = "splitters.sentence.SentenceSplitter"
    nltk_data_dir: str | None = None


class ChunkerConfig(ComponentConfig):
    type: str = "chunkers.recursive.RecursiveChunker"
    # All other fields (max_chunk_chars, overlap_chars, similarity_threshold, …)
    # are passed through via model_extra — specify them directly in the YAML
    # for whichever chunker type you are using.


class EmbedderConfig(ComponentConfig):
    type: str = "embedders.qwen.QwenEmbedder"
    # All other fields (model_name, device, batch_size, …) are passed through
    # via model_extra — specify them in the YAML for your chosen embedder.


class RetrieverConfig(ComponentConfig):
    type: str = "retrievers.faiss_retriever.FaissRetriever"
    # Extra fields (e.g. metric) passed through via model_extra.


class RerankerConfig(ComponentConfig):
    type: str = "rerankers.cross_encoder.CrossEncoderReranker"


class GeneratorConfig(ComponentConfig):
    type: str = "generators.gemini.GeminiGenerator"


class IterRetGenConfig(BaseModel):
    max_iterations: int = 3
    short_answer: bool = True  # True = brief intermediate answers; False = full answers
    intermediate_model_name: str = "gemini-2.5-flash"
    intermediate_max_output_tokens: int = 200


# ---------------------------------------------------------------------------
# Dataset config
# ---------------------------------------------------------------------------


class DatasetConfig(BaseModel):
    name: str
    # Path to documents.parquet from the released CanLegalRAGBench dataset
    path: str
    max_docs: int | None = None  # None = use all documents


# ---------------------------------------------------------------------------
# Evaluation config
# ---------------------------------------------------------------------------


class EvaluationConfig(BaseModel):
    queries_path: str
    k_values: list[int] = Field(default_factory=lambda: [3, 5, 10, 20, 50])
    rerank: bool = False
    generate: bool = False
    metrics: list[str] = Field(
        default_factory=lambda: ["recall_at_k", "precision_at_k", "mrr", "ndcg_at_k"]
    )


# ---------------------------------------------------------------------------
# Indexing config
# ---------------------------------------------------------------------------


class IndexingConfig(BaseModel):
    document_batch_size: int = 32
    embedding_batch_size: int = 32
    save_intermediate_every: int = 500  # docs
    # Use {index_id} to share the index across experiments with the same
    # dataset + chunker + embedder.  Falls back to {experiment_id} if overridden.
    output_dir: str = "runs/indexes/{index_id}"


# ---------------------------------------------------------------------------
# Logging config
# ---------------------------------------------------------------------------


class LoggingConfig(BaseModel):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_dir: str = "runs/{experiment_id}/logs"
    resource_monitor_interval: float = 30.0  # seconds; 0 = disabled


# ---------------------------------------------------------------------------
# Top-level experiment config
# ---------------------------------------------------------------------------


class ExperimentConfig(BaseModel):
    experiment_id: str
    description: str = ""
    seed: int = 42

    dataset: DatasetConfig
    splitter: SplitterConfig = Field(default_factory=SplitterConfig)
    chunker: ChunkerConfig = Field(default_factory=ChunkerConfig)
    embedder: EmbedderConfig = Field(default_factory=EmbedderConfig)
    retriever: RetrieverConfig = Field(default_factory=RetrieverConfig)
    reranker: RerankerConfig | None = None
    generator: GeneratorConfig | None = None
    iterretgen: IterRetGenConfig | None = None
    evaluation: EvaluationConfig | None = None
    indexing: IndexingConfig = Field(default_factory=IndexingConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @property
    def index_id(self) -> str:
        """
        A short, deterministic identifier for the (dataset, chunker, embedder) triple.

        Two experiments with the same dataset path, max_docs, chunker type + params,
        and embedder type + model_name will share the same index_id — and therefore
        the same on-disk index.  Changing any of those fields produces a new id.
        """
        key = {
            "dataset_path": self.dataset.path,
            "dataset_max_docs": self.dataset.max_docs,
            "chunker_type": self.chunker.type,
            "chunker_max_chunk_chars": self.chunker.max_chunk_chars,
            "chunker_overlap_chars": self.chunker.overlap_chars,
            "embedder_type": self.embedder.type,
            "embedder_model_name": self.embedder.model_name,
        }
        digest = hashlib.sha1(
            json.dumps(key, sort_keys=True).encode()
        ).hexdigest()[:10]
        # Human-readable prefix for easier debugging
        embedder_short = self.embedder.model_name.split("/")[-1].lower().replace("-", "_")
        chunker_short = self.chunker.type.split(".")[-1].lower().replace("chunker", "")
        return f"{embedder_short}__{chunker_short}{self.chunker.max_chunk_chars}__{digest}"

    @model_validator(mode="after")
    def _resolve_templates(self) -> "ExperimentConfig":
        """Replace {experiment_id} and {index_id} placeholders in path fields."""
        eid = self.experiment_id
        iid = self.index_id
        self.indexing.output_dir = self.indexing.output_dir.format(
            experiment_id=eid, index_id=iid
        )
        self.logging.log_dir = self.logging.log_dir.format(
            experiment_id=eid, index_id=iid
        )
        return self

    # ------------------------------------------------------------------
    # I/O helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        """
        Load an experiment config from a YAML file.

        The YAML may use ``base_config`` to inherit from a base file::

            base_config: ../../configs/base.yaml
            experiment_id: my_experiment
            embedder:
              model_name: Qwen/Qwen3-Embedding-8B

        Keys in the experiment file are deep-merged on top of the base.
        """
        path = Path(path)
        raw = yaml.safe_load(path.read_text())

        if "base_config" in raw:
            base_path = path.parent / raw.pop("base_config")
            base_raw = yaml.safe_load(base_path.read_text())
            raw = _deep_merge(base_raw, raw)

        return cls.model_validate(raw)

    def to_yaml(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.dump(self.model_dump(), sort_keys=False))


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base (override wins on conflicts)."""
    result = dict(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result
