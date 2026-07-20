"""Step 4: Normalize law areas and group query blocks for annotation.

Takes the query variations from step 3 and:
1. Normalizes the judge-assigned `law_area` labels into a single lowercase
   category per row (`cleaned_law_area`).
2. Reorders the blocks of variations (one block per seed query, 8 rows each by
   default) so that blocks with the same law area are adjacent — this lets a
   human annotator work through one area of law at a time. Rows within a block
   stay together and in order.
3. Writes the final annotation-ready file plus a per-block law-area index.

Note: the original run of this pipeline also repaired a generation artifact in
the step-3 output here (the base row of the second fact-variation block held
the seed query instead of the fact variation). That bug is fixed at the source
in this release, so this step is now only cleaning and grouping.

Usage:
    python 4_clean_and_group.py \
        --input-file outputs/query_variations/query_variations.json
"""

import argparse
from pathlib import Path

import pandas as pd


def clean_law_area(law_area: pd.Series) -> pd.Series:
    """Reduce raw judge output (may be a stringified list) to one lowercase label."""
    cleaned = law_area.str.lower()
    cleaned = cleaned.str.replace(r"[\[\]]", "", regex=True)
    cleaned = cleaned.str.replace("'", "")
    cleaned = cleaned.str.split(",").str[0]          # keep the first listed area
    cleaned = cleaned.str.split(" law").str[0]       # drop the trailing " law"
    return cleaned


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-in", "--input-file", type=Path, required=True, help="query_variations.json from step 3")
    parser.add_argument("--block-size", type=int, default=8, help="rows per seed-query block (2 fact variations x 4 rows)")
    parser.add_argument("--output-dir", type=Path, default=None, help="output directory (default: same folder as the input file)")
    args = parser.parse_args()

    df = pd.read_json(args.input_file, orient="records")
    if len(df) % args.block_size != 0:
        raise ValueError(
            f"Input has {len(df)} rows, which is not a multiple of the block size "
            f"({args.block_size}). Check that step 3 completed without dropped rows."
        )

    df["cleaned_law_area"] = clean_law_area(df["law_area"])

    # Group whole blocks by the law area of their first row; blocks stay intact.
    blocks = []
    for i in range(0, len(df), args.block_size):
        block = df.iloc[i:i + args.block_size]
        blocks.append((block["cleaned_law_area"].iloc[0], block.index[0], block))
    blocks.sort(key=lambda x: (x[0], x[1]))
    df = pd.concat([block for _, _, block in blocks], ignore_index=True)

    df = df.rename(columns={"index": "original_index"})
    df["index"] = df.index

    output_dir = args.output_dir if args.output_dir is not None else args.input_file.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "query_variations_cleaned.json"
    df.to_json(output_file, orient="records")
    print(f"Saved {len(df)} cleaned and grouped query variations to {output_file}")

    # Per-block law-area index: one row per seed query, for annotation planning.
    law_area_df = df.loc[df["index"] % args.block_size == 0, ["cleaned_law_area", "index"]]
    law_area_file = output_dir / "law_area_index.csv"
    law_area_df.to_csv(law_area_file, index=False)
    print(f"Saved per-block law-area index to {law_area_file}")
    print(df["cleaned_law_area"].value_counts())


if __name__ == "__main__":
    main()