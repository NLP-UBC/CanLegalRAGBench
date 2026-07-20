"""
Run retrieval (and optionally generation + judging) evaluation for one experiment.

Usage
-----
    # Retrieval-only
    python scripts/run_benchmark.py --config configs/experiments/qwen_recursive_8192.yaml

    # With generation + LLM judge
    python scripts/run_benchmark.py --config configs/experiments/qwen_recursive_8192.yaml \
        --generate --judge

Expects an already-built index at runs/<experiment_id>/index/ (run run_indexing.py first).
Writes results to runs/<experiment_id>/results/.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Load .env before anything else so API keys are available to all components.
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass  # python-dotenv not installed; fall back to shell environment

import pandas as pd
from tqdm import tqdm

from benchmark_rag.components.base import BudgetExceededError
from benchmark_rag.config.schemas import ExperimentConfig
from benchmark_rag.cost_logging import (
    DEFAULT_BENCHMARK_COST_CSV,
    append_cost_entry,
    collect_component_costs,
)
from benchmark_rag.evaluation.metrics import evaluate_retrieval, EvaluationResult, is_query_usable
from benchmark_rag.logging import setup_experiment_logging, get_logger
from benchmark_rag.pipeline.rag_pipeline import RAGPipeline


def _validate_api_keys(cfg: ExperimentConfig, args) -> None:
    """Fail fast if required API keys are missing for the configured components."""
    import os
    embedder_type = cfg.embedder.type.lower()
    generator_type = (cfg.generator.type.lower() if cfg.generator else "")
    reranker_type = (cfg.reranker.type.lower() if cfg.reranker else "")

    needs_isaacus = "kanon2" in embedder_type or "kanon2" in reranker_type
    if needs_isaacus and not os.environ.get("ISAACUS_API_KEY"):
        sys.exit("ERROR: ISAACUS_API_KEY is not set. Required for Kanon2Embedder / Kanon2Reranker.")

    needs_google = (
        "gemini" in embedder_type
        or "gemini" in generator_type
        or args.iterretgen
    )
    if needs_google and not (os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")):
        sys.exit("ERROR: GOOGLE_API_KEY or GEMINI_API_KEY is not set. Required for Gemini components.")


def load_queries(queries_path: str) -> list[dict]:
    """
    Load the benchmark queries (queries.json / queries.parquet from the
    released CanLegalRAGBench dataset).

    Schema (one item per query):
        query_id, query_text, answer, batch_id,
        ground_truth_citations (list[str]), province

    JSON files may be a single array or json-lines (the released format).
    """
    p = Path(queries_path)
    if p.suffix == ".json":
        text = p.read_text(encoding="utf-8").strip()
        if text.startswith("["):
            return json.loads(text)
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    elif p.suffix == ".parquet":
        df = pd.read_parquet(p)
        records = df.to_dict(orient="records")
        for r in records:
            gt = r.get("ground_truth_citations")
            if gt is not None and not isinstance(gt, list):
                r["ground_truth_citations"] = list(gt)
        return records
    raise ValueError(f"Unsupported query file format: {p.suffix}")


def _log_run_context(log, cfg: ExperimentConfig, config_source: str, args) -> None:
    """Log the full config and all relevant file paths for this benchmark run."""
    run_dir = Path(f"runs/{cfg.experiment_id}")
    index_dir = Path(cfg.indexing.output_dir)
    results_dir = run_dir / "results"

    log.info("=" * 60)
    log.info(f"EXPERIMENT : {cfg.experiment_id}")
    log.info(f"DESCRIPTION: {cfg.description}")
    log.info(f"SEED       : {cfg.seed}")
    log.info(f"CONFIG SRC : {config_source}")
    log.info(f"FLAGS      : generate={args.generate}  judge={args.judge}  iterretgen={args.iterretgen}")
    log.info(f"INDEX ID   : {cfg.index_id}")

    # --- Input paths ---
    log.info("--- inputs ---")
    log.info(f"  dataset.path    : {cfg.dataset.path}")
    log.info(f"  dataset.max_docs: {cfg.dataset.max_docs}")
    log.info(f"  queries_path    : {cfg.evaluation.queries_path}")
    log.info(f"  index dir       : {index_dir}")
    log.info(f"    index.faiss   : {index_dir}/index.faiss")
    log.info(f"    index.chunks  : {index_dir}/index.chunks.pkl")

    # --- Component config ---
    log.info("--- components ---")
    log.info(f"  embedder : {cfg.embedder.type}")
    log.info(f"    model_name  : {cfg.embedder.model_name}")
    log.info(f"    device      : {cfg.embedder.model_extra.get('device', 'N/A')}")
    log.info(f"  chunker  : {cfg.chunker.type}")
    log.info(f"    max_chunk_chars : {cfg.chunker.max_chunk_chars}")
    log.info(f"    overlap_chars   : {cfg.chunker.overlap_chars}")
    log.info(f"  retriever: {cfg.retriever.type}")
    log.info(f"    metric      : {cfg.retriever.model_extra.get('metric', 'cosine')}")
    if cfg.generator:
        log.info(f"  generator: {cfg.generator.type}")
        log.info(f"    model_name  : {cfg.generator.model_name}")

    # --- Evaluation config ---
    log.info("--- evaluation ---")
    log.info(f"  k_values : {cfg.evaluation.k_values}")
    log.info(f"  metrics  : {cfg.evaluation.metrics}")

    # --- Output paths ---
    log.info("--- outputs ---")
    log.info(f"  results dir      : {results_dir}")
    log.info(f"    metrics        : {results_dir}/metrics.json")
    log.info(f"    query results  : {results_dir}/query_results.jsonl")
    log.info(f"  log dir          : {cfg.logging.log_dir}")
    log.info(f"    human log      : {cfg.logging.log_dir}/{cfg.experiment_id}.log")
    log.info(f"    json log       : {cfg.logging.log_dir}/{cfg.experiment_id}.jsonl")
    log.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Evaluate a RAG experiment.")
    parser.add_argument("--config", required=True, help="Path to experiment YAML config")
    parser.add_argument("--generate", action="store_true", help="Run answer generation")
    parser.add_argument("--judge", action="store_true", help="Run LLM judge on generated answers")
    parser.add_argument("--iterretgen", action="store_true", help="Use IterRetGen pipeline (iterative retrieval augmented by intermediate generation)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip queries already in query_results.jsonl and append new results")
    args = parser.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)
    _validate_api_keys(cfg, args)

    if cfg.evaluation is None:
        print("No evaluation config found — nothing to do.")
        sys.exit(1)

    setup_experiment_logging(
        experiment_id=cfg.experiment_id,
        log_dir=cfg.logging.log_dir,
        level=cfg.logging.level,
        resource_monitor_interval=0,  # no background monitor during eval
    )
    log = get_logger(__name__)
    _log_run_context(log, cfg, config_source=args.config, args=args)

    results_dir = Path(f"runs/{cfg.experiment_id}/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    # --- Load queries ---
    # queries.json schema: query_id, query_text, user_answer, ground_truth_citations (list)
    queries = load_queries(cfg.evaluation.queries_path)
    log.info(f"Loaded {len(queries)} queries from {cfg.evaluation.queries_path}")

    # --- Build pipeline ---
    eval_cfg = cfg.evaluation
    is_hybrid = "hybrid" in cfg.retriever.type.lower()

    if args.iterretgen:
        from benchmark_rag.pipeline.iterretgen_pipeline import IterRetGenPipeline
        pipeline = IterRetGenPipeline.from_config(cfg)
    elif is_hybrid:
        from benchmark_rag.pipeline.hybrid_pipeline import HybridRAGPipeline
        pipeline = HybridRAGPipeline.from_config(cfg)
    else:
        pipeline = RAGPipeline.from_config(cfg)

    # --- Resume support: load previously completed query IDs ---
    completed_ids: set = set()
    rows = []
    if args.resume:
        results_file = results_dir / "query_results.jsonl"
        if results_file.exists():
            with open(results_file) as f:
                for line in f:
                    rec = json.loads(line)
                    # Only count as completed if it actually ran (has retrieved_ids)
                    if rec.get("retrieved_ids") or rec.get("saved_docs"):
                        completed_ids.add(rec["query_id"])
                        rows.append(rec)
            log.info(f"Resume: loaded {len(completed_ids)} completed queries from {results_file}")
        failed_file = results_dir / "failed_queries.json"
        if failed_file.exists():
            prev_failed = json.loads(failed_file.read_text())
            for fq in prev_failed:
                completed_ids.discard(fq["query_id"])
            log.info(f"Resume: {len(prev_failed)} previously failed queries will be retried")

    # --- Run queries ---
    # ground_truth_citations is a list[str] — a query may have multiple relevant docs
    all_retrieved: list[list[str]] = []
    all_relevant: list[set[str]] = []
    failed_queries: list[dict] = []

    # Populate metrics lists from resumed rows
    for rec in rows:
        all_retrieved.append(rec.get("retrieved_ids", []))
        all_relevant.append(set(rec.get("gold_citations", [])))

    t0 = time.perf_counter()
    total_queries = len(queries)
    checkpoint_interval = max(1, total_queries // 5)  # save every 20%
    queries_processed = 0

    for q in tqdm(queries, desc="Querying"):
        query_text = str(q.get("query_text", ""))
        province = q.get("province", "")
        if province:
            query_text = f"I am in {province}. {query_text}"
        gold_citations: set[str] = set(q.get("ground_truth_citations", []))
        if not query_text.strip() or not is_query_usable(q):
            continue

        if q.get("query_id") in completed_ids:
            continue

        try:
            result = pipeline.query(query_text, k=max(eval_cfg.k_values))
        except BudgetExceededError as exc:
            log.warning(
                "Budget exceeded at query %s — saving results and stopping. %s",
                q.get("query_id"), exc,
            )
            break
        except Exception as exc:
            query_id = q.get("query_id")
            log.error(
                "Query %s failed (%s: %s) — skipping. "
                "query_text=%r",
                query_id, type(exc).__name__, exc,
                query_text[:200],
            )
            failed_queries.append({
                "query_id": query_id,
                "query_text": query_text,
                "gold_citations": list(gold_citations),
                "error": f"{type(exc).__name__}: {exc}",
            })
            continue

        retrieved_ids = [c.doc_id for c in result.retrieved_chunks]

        all_retrieved.append(retrieved_ids)
        all_relevant.append(gold_citations)

        unique_retrieved_docs = list(dict.fromkeys(retrieved_ids))
        record = {
            "query_id": q.get("query_id"),
            "query_text": query_text,
            "gold_citations": list(gold_citations),
            "retrieved_ids": retrieved_ids,
            "retrieved_chunk_details": [
                {"doc_id": c.doc_id, "chunk_idx": c.chunk_idx, "score": round(c.score, 6)}
                for c in result.retrieved_chunks
            ],
            "num_unique_docs_retrieved": len(unique_retrieved_docs),
            "answer": result.answer,
        }
        if result.metadata:
            record["iterations"] = result.metadata.get("iterations")
            record["num_searches"] = len(result.metadata.get("searches_run", []))
            record["saved_docs"] = result.metadata.get("saved_docs")
        rows.append(record)

        queries_processed += 1
        if queries_processed % checkpoint_interval == 0:
            checkpoint_file = results_dir / "query_results.jsonl"
            with open(checkpoint_file, "w") as f:
                for rec in rows:
                    f.write(json.dumps(rec) + "\n")
            log.info(f"Checkpoint saved at {queries_processed}/{total_queries} queries")

    elapsed = time.perf_counter() - t0
    log.info(f"Queried {len(rows)} examples in {elapsed:.1f}s")
    if failed_queries:
        log.warning(f"{len(failed_queries)} query/queries failed — see failed_queries.json")
    if hasattr(pipeline, "log_usage_summary"):
        pipeline.log_usage_summary()
    if hasattr(pipeline, "generator") and pipeline.generator is not None and hasattr(pipeline.generator, "log_usage_summary"):
        pipeline.generator.log_usage_summary()

    # --- Compute metrics ---
    eval_result: EvaluationResult = evaluate_retrieval(
        experiment_id=cfg.experiment_id,
        retrieved_lists=all_retrieved,
        relevant_sets=all_relevant,
        k_values=eval_cfg.k_values,
        metric_names=eval_cfg.metrics,
    )

    # --- Optional: LLM judge ---
    judge = None
    if args.judge and cfg.generator is not None:
        from benchmark_rag.components.generators.gemini import GeminiJudge

        # Build a lookup: query_id → reference answer (written by the annotator)
        ref_by_id = {
            str(q.get("query_id")): str(q.get("answer", q.get("user_answer", "")))
            for q in queries
        }

        judge = GeminiJudge()
        judge_scores: dict[str, list[float]] = {}
        for rec in tqdm(rows, desc="Judging"):
            if not rec.get("answer"):
                continue
            ref = ref_by_id.get(str(rec.get("query_id")), "")
            scores = judge.judge(rec["query_text"], rec["answer"], ref)
            for k, v in scores.items():
                if k != "rationale":
                    judge_scores.setdefault(k, []).append(float(v))
        eval_result.judge_scores = {k: sum(v) / len(v) for k, v in judge_scores.items() if v}
        judge.log_usage_summary()

    # --- Save results ---
    results_file = results_dir / "query_results.jsonl"
    with open(results_file, "w") as f:
        for rec in rows:
            f.write(json.dumps(rec) + "\n")

    if failed_queries:
        failed_file = results_dir / "failed_queries.json"
        failed_file.write_text(json.dumps(failed_queries, indent=2))
        log.info(f"Saved {len(failed_queries)} failed query/queries to {failed_file}")

    # Aggregate retrieval stats
    doc_counts = [r["num_unique_docs_retrieved"] for r in rows]
    retrieval_stats = {
        "mean_docs_retrieved": round(sum(doc_counts) / len(doc_counts), 2) if doc_counts else 0,
        "median_docs_retrieved": round(sorted(doc_counts)[len(doc_counts) // 2], 2) if doc_counts else 0,
        "min_docs_retrieved": min(doc_counts) if doc_counts else 0,
        "max_docs_retrieved": max(doc_counts) if doc_counts else 0,
    }
    if any("iterations" in r for r in rows):
        iter_counts = [r["iterations"] for r in rows if r.get("iterations") is not None]
        retrieval_stats["mean_iterations"] = round(sum(iter_counts) / len(iter_counts), 2) if iter_counts else 0
        search_counts = [r.get("num_searches") or 0 for r in rows]
        retrieval_stats["mean_searches"] = round(sum(search_counts) / len(search_counts), 2) if search_counts else 0

    metrics_file = results_dir / "metrics.json"
    metrics_file.write_text(
        json.dumps(
            {
                "experiment_id": eval_result.experiment_id,
                "num_queries": eval_result.num_queries,
                "num_failed": len(failed_queries),
                "retrieval_stats": retrieval_stats,
                "scores": {m: dict(by_k) for m, by_k in eval_result.scores.items()},
                "judge_scores": eval_result.judge_scores,
            },
            indent=2,
        )
    )

    log.info(f"Results saved to {results_dir}")

    cost_breakdown = collect_component_costs(pipeline, judge)
    total, csv_path = append_cost_entry(
        DEFAULT_BENCHMARK_COST_CSV,
        experiment_id=cfg.experiment_id,
        cost_of_run_usd=cost_breakdown["total"],
        cost_breakdown=cost_breakdown,
    )
    log.info(
        f"Cost logged to {csv_path}: run=${cost_breakdown['total']:.6f} "
        f"(embed=${cost_breakdown['embedding']:.6f} "
        f"rerank=${cost_breakdown['reranker']:.6f} "
        f"gen=${cost_breakdown['generator']:.6f} "
        f"other=${cost_breakdown['other']:.6f}) "
        f"| total_so_far=${total:.6f}"
    )

    print(eval_result.summary())


if __name__ == "__main__":
    main()
