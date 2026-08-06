"""Step 2: Filter generated queries with an LLM judge.

Each (query, document) pair from step 1 is audited by `LegalJudge` (see
llm_judges/judge.py) against criteria for realism, retrievability, grounding,
and everyday-person relevance. Pairs judged 'Reject' are removed.

Outputs (all CSV, in outputs/<experiment-name>/):
- judged_validation_set_*        all accepted queries
- rejected_validation_set_*      rejected queries (kept for inspection)
- readable_judged_validation_set_*  accepted layperson queries with bulky
  columns dropped; for york, every other TCC query is also dropped to
  rebalance the 2x TCC oversampling from step 1

Usage:
    python filter_by_judge.py --experiment-name my_run --dataset york \
        --input-file outputs/my_run/validation_set_york.json
"""

import argparse
import os
from pathlib import Path

import dspy
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

from llm_judges.judge import LegalJudge

# Bulky or redundant columns dropped from the readable output (only those present are dropped).
READABLE_DROP_COLUMNS = [
    "document_date_en", "document_date_fr", "name_fr", "name_en", "upstream_license",
    "unofficial_text_en", "unofficial_text_fr", "url_en", "url_fr",
    "scraped_timestamp_fr", "scraped_timestamp_en", "context_snippet", "user_traits",
    "stitched_text",
]


def clean_text(text):
    """Undo JSON-style escaping so the judge sees the document as plain text."""
    if not isinstance(text, str):
        return text
    text = text.replace(r"\/", "/")
    try:
        text = text.encode("utf-8").decode("unicode_escape")
    except UnicodeError:
        pass
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-exp", "--experiment-name", required=True, help="name for this run; outputs go to outputs/<experiment-name>/")
    parser.add_argument("-d", "--dataset", required=True, choices=["york", "caseway"], help="document source of the input file")
    parser.add_argument("-in", "--input-file", type=Path, required=True, help="validation set JSON (lines) from step 1")
    parser.add_argument("--model", default="gemini/gemini-2.5-flash", help="LiteLLM model string for the judge")
    parser.add_argument("--max-cost", type=float, default=10.0, help="abort once judging cost exceeds this many dollars")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="base output directory")
    args = parser.parse_args()

    load_dotenv()
    lm = dspy.LM(args.model, api_key=os.getenv("GEMINI_API_KEY"))
    dspy.configure(lm=lm, track_usage=True)
    judge = LegalJudge(total_max_cost=args.max_cost)

    df = pd.read_json(args.input_file, orient="records", lines=True)
    print(f"loaded {len(df)} queries from {args.input_file}")

    text_key = "stitched_text" if args.dataset == "caseway" else "unofficial_text_en"
    accepted, rejected = [], []
    total_cost = 0.0

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Judging queries"):
        query = row["query"]
        if not isinstance(query, str) or query.strip() == "":
            print(f"Skipping empty or invalid query at index {idx}")
            continue

        document = clean_text(row[text_key])
        judge_result = judge(query=query, document_snippet=document).results[0]
        verdict = judge_result["final_verdict"]
        total_cost += judge_result.get("judge_cost", 0.0)

        if not isinstance(verdict, str) or ("Keep" in verdict and "Reject" in verdict):
            print(f"Skipping ambiguous verdict {verdict!r} for query at index {idx}: {query}")
            continue

        entry = row.to_dict()
        entry["judge_result"] = verdict
        if "Keep" in verdict:
            accepted.append(entry)
        elif "Reject" in verdict:
            rejected.append(entry)
        else:
            print(f"Skipping unexpected verdict {verdict!r} for query at index {idx}: {query}")

    print(f"Accepted {len(accepted)} queries, Rejected {len(rejected)} queries.")

    output_dir = args.output_dir / args.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    accepted_df = pd.DataFrame(accepted)
    accepted_file = output_dir / f"judged_validation_set_{args.dataset}.csv"
    accepted_df.to_csv(accepted_file, index=False)
    print(f"Saved {len(accepted_df)} accepted queries to {accepted_file}")

    rejected_df = pd.DataFrame(rejected)
    rejected_file = output_dir / f"rejected_validation_set_{args.dataset}.csv"
    rejected_df.to_csv(rejected_file, index=False)
    print(f"Saved {len(rejected_df)} rejected queries to {rejected_file}")

    # Readable subset: layperson queries only, bulky columns dropped.
    readable_df = accepted_df.drop(columns=[c for c in READABLE_DROP_COLUMNS if c in accepted_df.columns])
    readable_df = readable_df[readable_df["user_type"] == "layperson"]
    # Optional extra filter queries to
    # courts most relevant to everyday people (small claims, provincial, family, municipal).
    # if args.dataset == "caseway":
    #     relevant_courts = "abcj,abpc,bcpc,mbpc,nlpc,nssm,nsfc,nspc,nucj,oncj,onscsm,qccq,qccm,qctdp,skpc,yktc".split(",")
    #     readable_df = readable_df[readable_df["court"].isin(relevant_courts)]
    if args.dataset == "york":
        # Rebalance the 2x TCC oversampling from step 1 by dropping every other TCC query.
        tcc_indices = readable_df[readable_df["dataset"] == "TCC"].index
        readable_df = readable_df.drop(tcc_indices[::2])

    readable_file = output_dir / f"readable_judged_validation_set_{args.dataset}.csv"
    readable_df.to_csv(readable_file, index=False)
    print(f"Saved {len(readable_df)} readable accepted queries to {readable_file}")

    print(f"Total judging cost: ${total_cost:.4f}")


if __name__ == "__main__":
    main()