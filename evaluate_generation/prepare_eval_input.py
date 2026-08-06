"""Step 2: Join generated answers with the reference answers for judging.

Takes a retrieval run's query_results.jsonl (with answers from step 1) and the
benchmark queries file, and writes one json-lines record per answered query in
the format the ragas judge (step 3) expects:

    query_id, query_text, generated_answer, ground_truth_answer,
    condition, generator, retrieval_method

Usage
-----
    python prepare_eval_input.py \
        --results ../evaluate_retrieval/runs/qwen_recursive_8192_1k-docs/results/query_results.jsonl \
        --generator gemini-2.5-flash --retrieval-method qwen_recursive_8192
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_queries(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8").strip()
    if text.startswith("["):
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, required=True,
                        help="query_results.jsonl from a retrieval run with generated answers")
    parser.add_argument("--queries", type=Path,
                        default=Path(__file__).resolve().parent.parent / "evaluate_retrieval" / "data" / "queries.json",
                        help="benchmark queries file (for the reference answers)")
    parser.add_argument("--out", type=Path, default=Path("outputs/answers.jsonl"),
                        help="output json-lines file (default: outputs/answers.jsonl)")
    parser.add_argument("--condition", default="pipeline",
                        help="bookkeeping label, e.g. 'pipeline' or 'oracle' (default: pipeline)")
    parser.add_argument("--generator", default="",
                        help="bookkeeping label for the generator model used in step 1")
    parser.add_argument("--retrieval-method", default="",
                        help="bookkeeping label for the retrieval experiment")
    args = parser.parse_args()

    queries = load_queries(args.queries)
    ref_by_id = {q["query_id"]: q.get("answer", q.get("user_answer", "")) for q in queries}
    print(f"Loaded {len(ref_by_id)} reference answers from {args.queries}")

    with open(args.results) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    print(f"Loaded {len(rows)} retrieval results from {args.results}")

    out_rows = []
    missing_answer = 0
    missing_ref = 0
    for row in rows:
        answer = (row.get("answer") or "").strip()
        if not answer:
            missing_answer += 1
            continue
        ref = (ref_by_id.get(row.get("query_id")) or "").strip()
        if not ref:
            missing_ref += 1
            continue
        out_rows.append({
            "query_id": row.get("query_id"),
            "query_text": row.get("query_text", ""),
            "generated_answer": answer,
            "ground_truth_answer": ref,
            "condition": args.condition,
            "generator": args.generator,
            "retrieval_method": args.retrieval_method,
        })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Wrote {len(out_rows)} records to {args.out} "
          f"({missing_answer} skipped without answer, {missing_ref} without reference)")


if __name__ == "__main__":
    main()