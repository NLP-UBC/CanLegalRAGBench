"""Step 1: Generate answers for an existing retrieval run.

Loads a retrieval experiment's query_results.jsonl (which has retrieved chunk
details), looks up chunk text from the index, builds the generation context
(full documents when they fit the token budget, otherwise chunks + truncated
documents), and calls Gemini to generate answers. Answers are written back
into query_results.jsonl under the "answer" key.

Run a retrieval experiment in ../evaluate_retrieval first (its runs/ directory
and index are read from there).

Usage
-----
    python generate_answers.py \
        --config ../evaluate_retrieval/configs/experiments/qwen_recursive_8192.yaml

    # Override model or token budget:
    python generate_answers.py --config ... --model gemini-2.5-pro --token-budget 64000

    # Test on 5 queries:
    python generate_answers.py --config ... --max-queries 5

Requires: GOOGLE_API_KEY or GEMINI_API_KEY
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

# The retrieval framework (benchmark_rag) and all run artifacts live in ../evaluate_retrieval.
RETRIEVAL_ROOT = Path(__file__).resolve().parent.parent / "evaluate_retrieval"
sys.path.insert(0, str(RETRIEVAL_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv()
    load_dotenv(RETRIEVAL_ROOT / ".env")
except ImportError:
    pass

import pandas as pd
from tqdm import tqdm

from benchmark_rag.components.base import EmbeddedChunk, RetrievedChunk
from benchmark_rag.config.schemas import ExperimentConfig
from benchmark_rag.logging import setup_experiment_logging, get_logger
from benchmark_rag.prompts.answer_generator import ANSWER_SYSTEM_PROMPT

CHARS_PER_TOKEN = 4


def _token_estimate(text: str) -> int:
    return len(text) // CHARS_PER_TOKEN


def build_context(
    chunks: list[RetrievedChunk],
    doc_texts: dict[str, str],
    token_budget: int,
) -> tuple[str, dict]:
    """Build generator context from retrieved chunks and full documents.

    Uses full documents if they fit the token budget, otherwise includes
    chunks plus proportionally truncated documents.
    """
    doc_ids = list(dict.fromkeys(c.doc_id for c in chunks))
    full_docs = {did: doc_texts[did] for did in doc_ids if did in doc_texts}

    total_doc_tokens = sum(_token_estimate(t) for t in full_docs.values())

    if total_doc_tokens <= token_budget:
        parts = []
        for did in doc_ids:
            if did in full_docs:
                parts.append(f"=== Document: {did} ===\n{full_docs[did]}")
        context = "\n\n".join(parts)
        return context, {
            "context_mode": "full_documents",
            "num_docs": len(full_docs),
            "total_doc_tokens": total_doc_tokens,
            "token_budget": token_budget,
        }

    chunk_tokens = sum(_token_estimate(c.text) for c in chunks)
    remaining = token_budget - chunk_tokens

    meta = {
        "context_mode": "chunks_and_truncated_docs",
        "num_docs": len(full_docs),
        "num_chunks": len(chunks),
        "chunk_tokens": chunk_tokens,
        "total_doc_tokens": total_doc_tokens,
        "token_budget": token_budget,
    }

    if remaining <= 0:
        parts = []
        for i, c in enumerate(chunks, 1):
            parts.append(f"[{i}] ({c.doc_id})\n{c.text}")
        meta["context_mode"] = "chunks_only"
        meta["truncation_ratio"] = 0.0
        return "\n\n".join(parts), meta

    ratio = remaining / total_doc_tokens
    meta["truncation_ratio"] = round(ratio, 4)

    chunk_parts = []
    for i, c in enumerate(chunks, 1):
        chunk_parts.append(f"[{i}] ({c.doc_id})\n{c.text}")
    chunk_section = "\n\n".join(chunk_parts)

    doc_parts = []
    for did in doc_ids:
        if did not in full_docs:
            continue
        full_text = full_docs[did]
        truncated_chars = int(len(full_text) * ratio)
        doc_parts.append(f"=== Document: {did} (truncated) ===\n{full_text[:truncated_chars]}")
    doc_section = "\n\n".join(doc_parts)

    context = (
        "RETRIEVED PASSAGES:\n"
        f"{chunk_section}\n\n"
        "FULL DOCUMENT CONTEXT (truncated to fit token budget):\n"
        f"{doc_section}"
    )
    return context, meta


def main():
    parser = argparse.ArgumentParser(
        description="Generate answers for an existing retrieval run.",
    )
    parser.add_argument("--config", required=True,
                        help="Retrieval experiment YAML (finds index and results dir)")
    parser.add_argument("--model", default="gemini-2.5-flash",
                        help="Gemini model for generation (default: gemini-2.5-flash)")
    parser.add_argument("--token-budget", type=int, default=115_000,
                        help="Max context tokens (default: 115000)")
    parser.add_argument("--max-queries", type=int, default=None,
                        help="Limit number of queries (for testing)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Regenerate answers even for queries that already have one")
    args = parser.parse_args()

    if not (os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")):
        sys.exit("ERROR: set GOOGLE_API_KEY or GEMINI_API_KEY")

    cfg = ExperimentConfig.from_yaml(args.config)

    setup_experiment_logging(
        experiment_id=cfg.experiment_id,
        log_dir=str(RETRIEVAL_ROOT / cfg.logging.log_dir),
        level=cfg.logging.level,
        resource_monitor_interval=0,
    )
    log = get_logger(__name__)

    results_dir = RETRIEVAL_ROOT / "runs" / cfg.experiment_id / "results"
    results_file = results_dir / "query_results.jsonl"
    if not results_file.exists():
        sys.exit(f"ERROR: {results_file} not found. Run retrieval first.")

    # --- Load existing results ---
    with open(results_file) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    log.info(f"Loaded {len(rows)} results from {results_file}")

    already_have = sum(1 for r in rows if r.get("answer"))
    need_answer = sum(1 for r in rows if not r.get("answer"))
    log.info(f"  {already_have} have answers, {need_answer} need generation")

    has_chunk_details = any(r.get("retrieved_chunk_details") for r in rows)
    if not has_chunk_details:
        sys.exit(
            "ERROR: query_results.jsonl lacks retrieved_chunk_details. "
            "Re-run retrieval with run_benchmark.py to save chunk info."
        )

    if not args.overwrite and need_answer == 0:
        print("All queries already have answers. Use --overwrite to regenerate.")
        return

    # --- Load chunk index for text lookup ---
    chunks_path = RETRIEVAL_ROOT / cfg.indexing.output_dir / "index.chunks.pkl"
    if not chunks_path.exists():
        sys.exit(f"ERROR: {chunks_path} not found.")

    log.info(f"Loading chunks from {chunks_path} ...")
    with open(chunks_path, "rb") as f:
        all_chunks: list[EmbeddedChunk] = pickle.load(f)

    chunk_lookup: dict[tuple[str, int], EmbeddedChunk] = {
        (c.doc_id, c.chunk_idx): c for c in all_chunks
    }
    log.info(f"  {len(all_chunks)} chunks indexed")

    # --- Load full document texts for context building ---
    dataset_path = RETRIEVAL_ROOT / cfg.dataset.path
    log.info(f"Loading documents from {dataset_path} ...")
    doc_texts = dict(zip(
        *pd.read_parquet(dataset_path, columns=["citation", "text"]).values.T
    ))
    log.info(f"  {len(doc_texts)} documents loaded")

    # --- Set up generator ---
    from google import genai
    from google.genai import types
    from benchmark_rag.components.generators.gemini import _generate_with_retry

    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)

    total_in_tokens = 0
    total_out_tokens = 0
    total_cost = 0.0
    generated = 0

    _PRICING = {
        "gemini-2.5-flash": (0.30 / 1_000_000, 0.30 / 1_000_000),
        "gemini-2.5-pro":   (1.25 / 1_000_000, 10.0 / 1_000_000),
    }

    def _cost(model, in_t, out_t):
        for prefix, (ip, op) in _PRICING.items():
            if model.startswith(prefix):
                return in_t * ip + out_t * op
        return 0.0

    # --- Generate answers ---
    process_rows = rows[:args.max_queries] if args.max_queries else rows
    total_to_process = len(process_rows)
    checkpoint_interval = max(1, total_to_process // 5)  # save every 20%

    def _save_checkpoint():
        with open(results_file, "w") as f_out:
            for r in rows:
                f_out.write(json.dumps(r) + "\n")

    for i, row in enumerate(tqdm(process_rows, desc="Generating")):
        if row.get("answer") and not args.overwrite:
            continue

        chunk_details = row.get("retrieved_chunk_details", [])
        if not chunk_details:
            continue

        # Reconstruct RetrievedChunk objects from saved details
        context_chunks: list[RetrievedChunk] = []
        for cd in chunk_details:
            key = (cd["doc_id"], cd["chunk_idx"])
            ec = chunk_lookup.get(key)
            if ec is None:
                continue
            context_chunks.append(RetrievedChunk(
                text=ec.text,
                doc_id=ec.doc_id,
                chunk_idx=ec.chunk_idx,
                metadata=ec.metadata,
                embedding=None,
                score=cd.get("score", 0.0),
            ))

        if not context_chunks:
            continue

        query_text = row.get("query_text", "")
        context, ctx_meta = build_context(context_chunks, doc_texts, args.token_budget)

        try:
            prompt = f"Context:\n{context}\n\nQuestion: {query_text}"
            response = _generate_with_retry(
                client,
                model=args.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=ANSWER_SYSTEM_PROMPT,
                    temperature=0.0,
                    max_output_tokens=16384,
                ),
            )
            usage = response.usage_metadata
            in_t = usage.prompt_token_count or 0
            out_t = usage.candidates_token_count or 0
            total_in_tokens += in_t
            total_out_tokens += out_t
            total_cost += _cost(args.model, in_t, out_t)

            row["answer"] = response.text
            row["answer_context_meta"] = ctx_meta
            generated += 1
        except Exception as e:
            log.error(f"Failed for query_id={row.get('query_id')}: {e}")

        if (i + 1) % checkpoint_interval == 0:
            _save_checkpoint()
            log.info(f"Checkpoint saved at {i+1}/{total_to_process} ({100*(i+1)//total_to_process}%%)")

    log.info(f"Generated {generated} answers")
    log.info(
        f"Cost: model={args.model} in={total_in_tokens} out={total_out_tokens} "
        f"cost=${total_cost:.6f}"
    )

    # --- Write back ---
    with open(results_file, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    log.info(f"Updated {results_file}")


if __name__ == "__main__":
    main()