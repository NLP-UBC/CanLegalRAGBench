"""
BM25 sparse retrieval benchmark.

Loads chunks from an existing index directory (built by run_indexing.py),
builds a BM25 index in memory, runs all queries, and reports retrieval metrics.

Usage
-----
    python scripts/run_bm25_benchmark.py --config configs/experiments/bm25_okapi_recursive_4096.yaml

Requires rank-bm25:
    pip install rank-bm25
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass

import pandas as pd
from tqdm import tqdm

from benchmark_rag.components.retrievers.bm25_retriever import BM25Retriever
from benchmark_rag.config.schemas import ExperimentConfig
from benchmark_rag.evaluation.metrics import evaluate_retrieval, is_query_usable
from benchmark_rag.logging import setup_experiment_logging, get_logger


def load_queries(queries_path: str) -> list[dict]:
    """Load benchmark queries; JSON may be a single array or json-lines (the released format)."""
    p = Path(queries_path)
    if p.suffix == ".json":
        text = p.read_text(encoding="utf-8").strip()
        if text.startswith("["):
            return json.loads(text)
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    elif p.suffix == ".parquet":
        records = pd.read_parquet(p).to_dict(orient="records")
        for r in records:
            gt = r.get("ground_truth_citations")
            if gt is not None and not isinstance(gt, list):
                r["ground_truth_citations"] = list(gt)
        return records
    raise ValueError(f"Unsupported query file format: {p.suffix}")


def _load_documents(cfg: ExperimentConfig):
    """Load Document objects from the dataset parquet (same logic as run_indexing.py)."""
    from benchmark_rag.components.base import Document
    import json as _json

    dataset_path = Path(cfg.dataset.path)
    max_docs = cfg.dataset.max_docs
    df = pd.read_parquet(dataset_path)
    if max_docs:
        df = df.head(max_docs)

    documents = []
    for _, row in df.iterrows():
        text = row.get("text", "")
        if not text or not str(text).strip():
            continue
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
        documents.append(Document(
            doc_id=str(row.get("citation", row.name)),
            text=str(text),
            metadata=meta,
        ))
    return documents


def main():
    parser = argparse.ArgumentParser(description="BM25 sparse retrieval benchmark.")
    parser.add_argument("--config", required=True, help="Path to experiment YAML config")
    parser.add_argument("--k1", type=float, default=None, help="BM25 k1 parameter (overrides config)")
    parser.add_argument("--b",  type=float, default=None, help="BM25 b parameter (overrides config)")
    parser.add_argument("--doc-level", action="store_true",
                        help="Index full documents instead of chunks (no FAISS index needed)")
    parser.add_argument("--variant", choices=["L", "okapi"], default=None,
                        help="BM25 variant: 'L' (default) or 'okapi'")
    args = parser.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)

    # k1 / b / variant: CLI arg wins; fall back to retriever config; then defaults
    k1 = args.k1 if args.k1 is not None else float(cfg.retriever.model_extra.get("k1", 1.5))
    b  = args.b  if args.b  is not None else float(cfg.retriever.model_extra.get("b",  0.75))
    variant = args.variant or cfg.retriever.model_extra.get("variant", "L")

    setup_experiment_logging(
        experiment_id=cfg.experiment_id,
        log_dir=cfg.logging.log_dir,
        level=cfg.logging.level,
        resource_monitor_interval=0,
    )
    log = get_logger(__name__)

    log.info("=" * 60)
    log.info(f"EXPERIMENT : {cfg.experiment_id}")
    log.info(f"DESCRIPTION: {cfg.description}")
    log.info(f"BM25 params: variant={variant}  k1={k1}  b={b}  doc_level={args.doc_level}")
    log.info(f"INDEX ID   : {cfg.index_id}")
    log.info("=" * 60)

    if args.doc_level:
        # --- Document-level BM25: load documents from parquet, no FAISS needed ---
        from benchmark_rag.components.base import EmbeddedChunk
        documents = _load_documents(cfg)
        log.info(f"Loaded {len(documents)} documents from {cfg.dataset.path}")
        chunks = [
            EmbeddedChunk(text=d.text, doc_id=d.doc_id, chunk_idx=0, metadata=d.metadata)
            for d in documents
        ]
        log.info(f"Building document-level BM25 index (variant={variant}, k1={k1}, b={b}) ...")
        retriever = BM25Retriever(variant=variant, k1=k1, b=b, index_level="document")
        retriever.build_index(chunks, documents=documents)
    else:
        # --- Chunk-level BM25: load chunks from existing FAISS index ---
        chunks_path = Path(cfg.indexing.output_dir) / "index.chunks.pkl"
        if not chunks_path.exists():
            log.error(f"Chunks file not found at {chunks_path}. Run run_indexing.py first.")
            sys.exit(1)
        log.info(f"Loading chunks from {chunks_path}")
        with open(chunks_path, "rb") as f:
            chunks = pickle.load(f)
        log.info(f"  {len(chunks)} chunks loaded.")
        log.info(f"Building BM25 index (variant={variant}, k1={k1}, b={b}) ...")
        retriever = BM25Retriever(variant=variant, k1=k1, b=b)
        retriever.build_index(chunks)

    log.info("  Done.")

    # --- Load queries ---
    queries = load_queries(cfg.evaluation.queries_path)
    log.info(f"Loaded {len(queries)} queries from {cfg.evaluation.queries_path}")

    # --- Run queries ---
    k_values = cfg.evaluation.k_values
    max_k = max(k_values)

    all_retrieved: list[list[str]] = []
    all_relevant:  list[set[str]] = []
    rows = []
    t0 = time.perf_counter()

    for q in tqdm(queries, desc="Querying (BM25)"):
        query_text = str(q.get("query_text", ""))
        province = q.get("province", "")
        if province:
            query_text = f"I am in {province}. {query_text}"
        gold_citations: set[str] = set(q.get("ground_truth_citations", []))
        if not query_text.strip() or not is_query_usable(q):
            continue

        results = retriever.retrieve_text(query_text, k=max_k)
        retrieved_ids = [c.doc_id for c in results]

        all_retrieved.append(retrieved_ids)
        all_relevant.append(gold_citations)
        rows.append({
            "query_id":       q.get("query_id"),
            "query_text":     query_text,
            "gold_citations": list(gold_citations),
            "retrieved_ids":  retrieved_ids,
            "answer":         None,
        })

    elapsed = time.perf_counter() - t0
    log.info(f"Queried {len(rows)} examples in {elapsed:.1f}s")

    # --- Compute metrics ---
    eval_result = evaluate_retrieval(
        experiment_id=cfg.experiment_id,
        retrieved_lists=all_retrieved,
        relevant_sets=all_relevant,
        k_values=k_values,
        metric_names=cfg.evaluation.metrics,
    )

    # --- Save results ---
    results_dir = Path(f"runs/{cfg.experiment_id}/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    results_file = results_dir / "query_results.jsonl"
    with open(results_file, "w") as f:
        for rec in rows:
            f.write(json.dumps(rec) + "\n")

    metrics_file = results_dir / "metrics.json"
    metrics_file.write_text(json.dumps(
        {
            "experiment_id": eval_result.experiment_id,
            "num_queries":   eval_result.num_queries,
            "scores":        {m: dict(by_k) for m, by_k in eval_result.scores.items()},
            "judge_scores":  eval_result.judge_scores,
        },
        indent=2,
    ))

    log.info(f"Results saved to {results_dir}")
    print(eval_result.summary())


if __name__ == "__main__":
    main()
