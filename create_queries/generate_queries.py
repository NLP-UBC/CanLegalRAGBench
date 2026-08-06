"""Step 1: Generate candidate queries from court decisions.

For each sampled document, generates one query per persona (layperson,
legal associate) with a randomly sampled Big Five trait profile and target
section (see prompts/create_prompts.py).

Two document sources are supported:
- york:    the A2AJ Canadian Legal Data dataset (SST, CHRT, TCC, BCSC courts),
           read as parquet from data/canadian-case-law/{COURT}/train.parquet
- caseway: a JSON list of provincial/small-claims court decisions
           (see data/README.md for the expected format)

English only.

Usage:
    python generate_queries.py --experiment-name my_run --dataset york
    python generate_queries.py --experiment-name my_run --dataset caseway \
        --caseway-file data/filtered_caseway_data.json
"""

import argparse
import os
from pathlib import Path

import dspy
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

from prompts.create_prompts import DocumentQueryGenerator

YORK_COURTS = ["SST", "CHRT", "TCC", "BCSC"]


def load_york_documents(data_dir: Path) -> pd.DataFrame:
    """Load the A2AJ Canadian Legal Data court parquet files into one dataframe."""
    dfs = []
    for court in YORK_COURTS:
        parquet_path = data_dir / court / "train.parquet"
        if not parquet_path.exists():
            raise FileNotFoundError(
                f"Missing {parquet_path}. Download the '{court}' subset of the "
                f"A2AJ Canadian Legal Data dataset (see data/README.md)."
            )
        dfs.append(pd.read_parquet(parquet_path))
    return pd.concat(dfs, ignore_index=True)


def sample_york_documents(df: pd.DataFrame, num_samples: int) -> pd.DataFrame:
    """Filter to interquartile document lengths, then sample per court.

    TCC is oversampled 2x because many TCC decisions are later rejected by the
    judge's tax-court relevance criterion (step 2), so extra candidates are needed.
    """
    df = df.dropna(subset=["unofficial_text_en"]).reset_index(drop=True)
    lengths = df["unofficial_text_en"].apply(lambda x: len(x.split()))
    q1, q3 = lengths.quantile(0.25), lengths.quantile(0.75)
    df = df[(lengths >= q1) & (lengths <= q3)].reset_index(drop=True)
    print(f"filtered to {len(df)} documents between Q1 ({q1:.0f}) and Q3 ({q3:.0f}) word counts")

    per_court = num_samples // len(YORK_COURTS)
    sampled = []
    for court in YORK_COURTS:
        court_docs = df[df["dataset"] == court]
        n = per_court * 2 if court == "TCC" else per_court
        sampled.append(court_docs.sample(n=min(len(court_docs), n), random_state=42))
    df = pd.concat(sampled, ignore_index=True)

    n = min(num_samples, len(df))
    return df.sample(n=n, random_state=42).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-exp", "--experiment-name", required=True, help="name for this run; outputs go to outputs/<experiment-name>/")
    parser.add_argument("-d", "--dataset", required=True, choices=["york", "caseway"], help="document source")
    parser.add_argument("--data-dir", type=Path, default=Path("data/canadian-case-law"), help="root of the A2AJ court parquet folders (york only)")
    parser.add_argument("--caseway-file", type=Path, default=Path("data/filtered_caseway_data.json"), help="JSON list of caseway documents (caseway only)")
    parser.add_argument("--num-samples", type=int, default=200, help="number of documents to sample (york only; caseway uses all documents)")
    parser.add_argument("--model", default="gemini/gemini-2.5-flash", help="LiteLLM model string for query generation")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="base output directory")
    args = parser.parse_args()

    load_dotenv()
    lm = dspy.LM(args.model, api_key=os.getenv("GEMINI_API_KEY"))
    dspy.configure(lm=lm, track_usage=True)

    if args.dataset == "york":
        df = load_york_documents(args.data_dir)
        print(f"loaded {len(df)} YORK documents")
        sampled_documents = sample_york_documents(df, args.num_samples)
        text_column = "unofficial_text_en"
        readable_columns = ["citation_en", "user_type", "target_section", "query"]
    else:
        # Caseway documents are expected to already be filtered to interquartile lengths.
        sampled_documents = pd.read_json(args.caseway_file).reset_index(drop=True)
        print(f"using all {len(sampled_documents)} CASEWAY documents")
        text_column = "stitched_text"
        readable_columns = ["url", "user_type", "target_section", "query"]

    print(f"sampled {len(sampled_documents)} documents")

    query_generator = DocumentQueryGenerator()
    query_data = []
    for idx, row in tqdm(sampled_documents.iterrows(), total=len(sampled_documents), desc="Generating queries"):
        doc = row[text_column]
        if not isinstance(doc, str) or len(doc.strip()) == 0:
            print(f"Skipping document at index {idx} due to no english text.")
            continue
        for result_dict in query_generator(decision_chunk=doc).results:
            new_row = row.to_dict()
            new_row.update(result_dict)
            query_data.append(new_row)

    final_df = pd.DataFrame(query_data)
    print(f"generated {len(final_df)} queries")
    print(f"total generation cost: ${final_df['cost'].sum():.4f}")

    output_dir = args.output_dir / args.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"validation_set_{args.dataset}.json"
    final_df.to_json(output_file, orient="records", lines=True)

    readable_file = output_dir / f"readable_validation_set_{args.dataset}.json"
    final_df[readable_columns].to_json(readable_file, orient="records", lines=True)

    print(f"Saved validation set ({len(sampled_documents)} docs, {len(final_df)} queries) to {output_file}")
    print(f"Saved readable validation set to {readable_file}")


if __name__ == "__main__":
    main()