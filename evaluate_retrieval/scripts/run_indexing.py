"""
Index a document corpus for a single experiment config.

Usage
-----
    python scripts/run_indexing.py --config configs/experiments/qwen_recursive_8192.yaml

The script:
  1. Loads the ExperimentConfig from YAML
  2. Sets up experiment logging
  3. Loads documents from the dataset path
  4. Runs IndexingPipeline → produces a FAISS index in runs/<experiment_id>/index/
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the package importable when run as a script from the repo root.
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load .env before anything else so API keys are available to all components.
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass  # python-dotenv not installed; fall back to shell environment

from benchmark_rag.config.schemas import ExperimentConfig
from benchmark_rag.cost_logging import (
    DEFAULT_INDEXING_COST_CSV,
    append_cost_entry,
    sum_component_costs,
)
from benchmark_rag.logging import setup_experiment_logging, get_logger
from benchmark_rag.pipeline.indexing_pipeline import IndexingPipeline


def _validate_api_keys(cfg: ExperimentConfig) -> None:
    """Fail fast if required API keys are missing for the configured components."""
    import os
    embedder_type = cfg.embedder.type.lower()

    if "kanon2" in embedder_type and not os.environ.get("ISAACUS_API_KEY"):
        sys.exit("ERROR: ISAACUS_API_KEY is not set. Required for Kanon2Embedder.")

    if "gemini" in embedder_type and not (os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")):
        sys.exit("ERROR: GOOGLE_API_KEY or GEMINI_API_KEY is not set. Required for GeminiEmbedder.")


def load_documents(cfg: ExperimentConfig):
    """
    Load documents from the dataset path specified in the config.

    Expects the dataset to be the released documents.parquet (see data/README.md)
    or a directory of .txt files.  Adapt this function to your actual data format.
    """
    from benchmark_rag.components.base import Document
    import pandas as pd

    dataset_path = Path(cfg.dataset.path)
    max_docs = cfg.dataset.max_docs

    documents: list[Document] = []

    if dataset_path.is_dir():
        # Directory of .txt files
        txt_files = sorted(dataset_path.glob("*.txt"))
        if max_docs:
            txt_files = txt_files[:max_docs]
        for f in txt_files:
            documents.append(
                Document(
                    doc_id=f.stem,
                    text=f.read_text(encoding="utf-8"),
                    metadata={"source_file": str(f)},
                )
            )
    elif dataset_path.suffix == ".parquet":
        import json as _json
        df = pd.read_parquet(dataset_path)
        if max_docs:
            df = df.head(max_docs)
        # Schema of the released documents.parquet:
        #   citation, name, original_source, year, text, url, upstream_license,
        #   is_ground_truth, dataset_source, ground_truth_query_ids (list)
        for _, row in df.iterrows():
            text = row.get("text", "")
            if not text or not str(text).strip():
                continue
            # Deserialise JSON-encoded list columns back to Python lists for metadata
            meta = {}
            for k, v in row.items():
                if k == "text":
                    continue
                if k in ("ground_truth_query_ids", "ground_truth_query_texts", "snippets"):
                    try:
                        meta[k] = _json.loads(v) if isinstance(v, str) else v
                    except Exception:
                        meta[k] = v
                else:
                    meta[k] = v
            documents.append(
                Document(
                    doc_id=str(row.get("citation", row.name)),
                    text=str(text),
                    metadata=meta,
                )
            )
    else:
        raise ValueError(
            f"Unsupported dataset path: {dataset_path}. "
            "Expected a directory of .txt files or a .parquet file."
        )

    return documents


def _log_run_context(log, cfg: ExperimentConfig, config_source: str) -> None:
    """Log the full config and all relevant file paths, and save config.json to the run dir."""
    import json

    run_dir = Path(cfg.indexing.output_dir).parent
    run_dir.mkdir(parents=True, exist_ok=True)

    # --- Save config snapshot ---
    config_snapshot = cfg.model_dump()
    config_out = run_dir / "config.json"
    config_out.write_text(json.dumps(config_snapshot, indent=2, default=str))

    # --- Log header ---
    log.info("=" * 60)
    log.info(f"EXPERIMENT : {cfg.experiment_id}")
    log.info(f"DESCRIPTION: {cfg.description}")
    log.info(f"SEED       : {cfg.seed}")
    log.info(f"CONFIG SRC : {config_source}")
    log.info(f"CONFIG SNAP: {config_out}")
    log.info(f"INDEX ID   : {cfg.index_id}")

    # --- Input paths ---
    log.info("--- inputs ---")
    log.info(f"  dataset.path    : {cfg.dataset.path}")
    log.info(f"  dataset.max_docs: {cfg.dataset.max_docs}")

    # --- Component config ---
    log.info("--- components ---")
    log.info(f"  chunker  : {cfg.chunker.type}")
    log.info(f"    max_chunk_chars : {cfg.chunker.max_chunk_chars}")
    log.info(f"    overlap_chars   : {cfg.chunker.overlap_chars}")
    log.info(f"  embedder : {cfg.embedder.type}")
    log.info(f"    model_name      : {cfg.embedder.model_name}")
    log.info(f"    device          : {cfg.embedder.model_extra.get('device', 'N/A')}")
    log.info(f"    batch_size      : {cfg.embedder.batch_size}")
    log.info(f"  retriever: {cfg.retriever.type}")
    log.info(f"    metric          : {cfg.retriever.model_extra.get('metric', 'cosine')}")

    # --- Output paths ---
    log.info("--- outputs ---")
    log.info(f"  index dir        : {cfg.indexing.output_dir}")
    log.info(f"    index.faiss    : {cfg.indexing.output_dir}/index.faiss")
    log.info(f"    index.chunks   : {cfg.indexing.output_dir}/index.chunks.pkl")
    log.info(f"    chunk metadata : {cfg.indexing.output_dir}/chunks_metadata.parquet")
    log.info(f"  log dir          : {cfg.logging.log_dir}")
    log.info(f"    human log      : {cfg.logging.log_dir}/{cfg.experiment_id}.log")
    log.info(f"    json log       : {cfg.logging.log_dir}/{cfg.experiment_id}.jsonl")
    log.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Run indexing pipeline for one experiment.")
    parser.add_argument("--config", required=True, help="Path to experiment YAML config")
    parser.add_argument(
        "--force-reindex",
        action="store_true",
        help="Rebuild the index even if one already exists for this dataset+chunker+embedder",
    )
    args = parser.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)
    _validate_api_keys(cfg)

    setup_experiment_logging(
        experiment_id=cfg.experiment_id,
        log_dir=cfg.logging.log_dir,
        level=cfg.logging.level,
        resource_monitor_interval=cfg.logging.resource_monitor_interval,
    )
    log = get_logger(__name__)
    _log_run_context(log, cfg, config_source=args.config)
    log.info(f"Index ID: {cfg.index_id}")

    documents = load_documents(cfg)
    log.info(f"Loaded {len(documents)} documents from {cfg.dataset.path}")

    pipeline = IndexingPipeline(cfg, force_reindex=args.force_reindex)
    try:
        output_dir = pipeline.run(documents)
    except Exception as exc:
        from benchmark_rag.components.base import BudgetExceededError
        if isinstance(exc, BudgetExceededError):
            log.warning(f"Budget exceeded during indexing — saving partial results. {exc}")
        else:
            raise
    else:
        log.info(f"Indexing complete. Output: {output_dir}")

    run_cost = sum_component_costs(pipeline)
    total, csv_path = append_cost_entry(
        DEFAULT_INDEXING_COST_CSV,
        experiment_id=cfg.experiment_id,
        cost_of_run_usd=run_cost,
    )
    log.info(
        f"Cost logged to {csv_path}: run=${run_cost:.6f} "
        f"| total_so_far=${total:.6f}"
    )


if __name__ == "__main__":
    main()
