"""Build the combined retrieval corpus and query set.

Combines three document sources into one deduplicated corpus:
  1. Ground-truth documents & queries — the human-annotated export
     (data/annotated_queries.json; private, see data/README.md)
  2. Caseway stitched documents       — distractors (private, see data/README.md)
  3. Stratified random sample from the A2AJ canadian-case-law corpus
     (BCSC, CHRT, SST, TCC)           — distractors (public)

Priority when deduplicating: ground truth > caseway > corpus sample.

Output (written to --output_dir):
  test_dataset.parquet  — all documents in the normalized schema
  test_dataset.json     — same data as JSON
  queries.json          — query metadata with ground-truth citation lists
  plus audit files: unparseable_court.json, unparseable_year.json,
  imputations.json, near_duplicate_char_counts.json

This script documents the process that produced the released benchmark; it is
not intended to be re-run, because the annotated input file is not public. The
built-and-released dataset is available at
https://huggingface.co/datasets/UBC-VL/CanLegalRAGBench

Usage
-----
    python build_test_dataset.py --samples_per_court 50 --output_dir outputs/test_dataset/

    # Skip caseway docs:
    python build_test_dataset.py --caseway_path ""
"""

import argparse
import hashlib
import json
import os
import random
import re

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

DEFAULT_COURTS = ["BCSC", "CHRT", "SST", "TCC"]
DEFAULT_SAMPLES_PER_COURT = 50
_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
DEFAULT_GROUND_TRUTH_PATH = os.path.join(_DATA_DIR, "annotated_queries.json")
DEFAULT_CASEWAY_PATH = os.path.join(_DATA_DIR, "filtered_caseway_data.json")
DEFAULT_CORPUS_DIR = os.path.join(_DATA_DIR, "canadian-case-law")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Statute / regulation detection — used to label missing-court documents.
# Matches common Canadian legislative citation prefixes (RSC, SC, RSO, SOR/, etc.)
# and keywords (Act, Code, Constitution, Rules).
_STATUTE_RE = re.compile(
    r"(?:^|\b)(?:R?S[A-Z]{1,4}\b|SOR/|SI/|[A-Z]+REG/|RRO\b|YOIC/|NLR/|OREG/|CQLR\b|RLRQ\b)"
    r"|\b(?i:Act|Code|Constitution|Rules)\b"
)

# Standard neutral citation: "2024 BCSC 123" — court is ALL-CAPS, number follows
_NEUTRAL_CITATION_RE = re.compile(r"\b(((?:19|20)\d{2})\s+([A-Z]{2,})\s+(\d+))\b")
# For court-only extraction from a neutral citation (used in _parse_court_from_citation)
_YEAR_COURT_RE = re.compile(r"\b(?:19|20)\d{2}[\s,]+([A-Z]{2,})\b")
# Bracketed year: "[1974] SCR 429" or "[2007] 1 SCR 429"
_BRACKET_YEAR_RE = re.compile(r"\[(?:19|20)\d{2}\]\s+(?:\d+\s+)?([A-Z]{2,})\b")
# CanLII format: "2011 CanLII 75710 (FC)" or "1999 CanLII 8630 (NL PC)"
_CANLII_RE = re.compile(r"CanLII\s+\d+\s+\(([^)]+)\)")
# CanLII citation as a whole: "2000 CanLII 15292 (FC)"
_CANLII_FULL_RE = re.compile(r"\b((?:19|20)\d{2}\s+CanLII\s+\d+(?:\s*\([^)]+\))?)")
# Caseway titles where "CanLII" keyword was stripped: "2012 48211 (NL SC) Case Name"
_CANLII_NO_KEYWORD_RE = re.compile(r"^((?:19|20)\d{2})\s+(\d+)\s+\(([^)]+)\)")
# Four-digit year anywhere in a citation
_YEAR_RE = re.compile(r"\b((?:19|20)\d{2})\b")


def _parse_court_from_citation(citation: str) -> str:
    """
    Extract court abbreviation from a citation string.

    Handles:
      - Neutral citations: '2024 BCSC 123' or 'R. v. Smith, 2024 BCSC 123'
      - Bracketed year:    '[2007] 1 SCR 429' or '[1974] SCJ No 95'
      - CanLII format:     '2011 CanLII 75710 (FC)'
      - Statute refs and Carswell citations return ''.
    """
    match = _YEAR_COURT_RE.search(citation)
    if match:
        return match.group(1)
    match = _BRACKET_YEAR_RE.search(citation)
    if match:
        return match.group(1)
    match = _CANLII_RE.search(citation)
    if match:
        return match.group(1).strip()
    return ""


def _parse_year_from_citation(citation: str) -> int | None:
    """Extract the four-digit year from a citation string, or None if not found."""
    match = _YEAR_RE.search(citation)
    return int(match.group(1)) if match else None


def _str(val) -> str:
    """Coerce a parquet scalar to str, handling None and float NaN.

    Parquet string columns with missing values come through as float('nan')
    when rows are accessed via .to_dict(), which is truthy and therefore
    not caught by the usual `val or ''` guard.
    """
    if isinstance(val, str):
        return val
    if val is None or (isinstance(val, float) and val != val):  # NaN: val != val
        return ""
    return str(val)


def _normalize_statute_ref(s: str) -> str:
    """Normalize punctuation in statute citations so variants map to the same key.

    Handles the common pattern where annotators write "S.C. 1996, c. 23"
    while another entry uses "SC 1996, c 23" — same statute, different style.

    Rule: remove dots that immediately follow a letter and precede another
    letter, a space, or a comma.  This covers abbreviation dots (S.C. → SC,
    R.S.C. → RSC, c. → c) without touching decimal numbers (e.g. s 27.1)
    or ellipses.
    """
    s = re.sub(r"(?<=[A-Za-z])\.(?=[A-Za-z\s,])", "", s)
    return " ".join(s.split())


def _normalize_citation(raw: str) -> str:
    """
    Extract the canonical citation from a messy annotation string.

    Annotators often paste full CanLII strings like:
        'Bernard v. Canada, 2014 SCC 13 (CanLII), [2014] 1 SCR 227, <https://...>'
    This function extracts just the primary neutral citation ('2014 SCC 13').

    Priority:
      1. Neutral citation  YYYY COURT NNN  (most authoritative)
      2. CanLII full form  YYYY CanLII NNNNN (COURT)
      3. Statute / other unrecognized form — punctuation-normalized so that
         'S.C. 1996, c. 23' and 'SC 1996, c 23' resolve to the same key.
    """
    raw = raw.strip()
    match = _NEUTRAL_CITATION_RE.search(raw)
    if match:
        return match.group(1)
    match = _CANLII_FULL_RE.search(raw)
    if match:
        return match.group(1).strip()
    return _normalize_statute_ref(raw)


def _load_ground_truth(path: str) -> tuple[list[dict], list[dict]]:
    """
    Parse the human-annotated query export (see data/README.md for the schema).

    Returns
    -------
    gt_docs : list[dict]
        One entry per unique citation, already in the normalized schema.
        ``snippets_by_query`` maps query_id → list[snippet_text] so that
        snippets remain attributable to the query that surfaced them.
    queries : list[dict]
        One entry per annotated query (query-level metadata + GT citation list).
        ``ground_truth_snippets`` maps citation → list[snippet_text] for
        the snippets that were annotated as relevant to *this* query.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # citation → normalized doc dict  (one doc can appear in multiple queries)
    gt_docs: dict[str, dict] = {}
    queries: list[dict] = []

    for qa in data:
        query_id = qa["query_id"]
        query_text = qa["query_text"]
        user_answer = qa.get("user_answer", "")
        custom_instruction = qa.get("custom_instruction", "")
        province = qa.get("province", "")
        batch_id = qa.get("batch_id", -1)
        query_citations: list[str] = []
        # citation → snippets relevant to THIS query (goes into the query record)
        query_snippets: dict[str, list[str]] = {}
        # Track citations already seen in this query to prevent intra-query duplicates
        seen_in_query: set[str] = set()

        for doc in qa.get("documents", []):
            citation = _normalize_citation((doc.get("citation") or "").strip())
            if not citation:
                continue
            if citation in seen_in_query:
                continue
            seen_in_query.add(citation)
            query_citations.append(citation)
            snippets = [
                s["text"]
                for s in doc.get("snippets", [])
                if s.get("text")
            ]
            query_snippets[citation] = snippets

            if citation not in gt_docs:
                text = doc.get("full_text") or ""
                if not text and snippets:
                    text = "\n\n".join(snippets)
                gt_docs[citation] = {
                    "citation": citation,
                    "citation2": "",
                    "name": doc.get("title", ""),
                    "court": _parse_court_from_citation(citation),
                    "year": _parse_year_from_citation(citation),
                    "char_count": len(text),
                    "text": text,
                    "url": doc.get("source_url", ""),
                    "upstream_license": "",
                    "document_date": None,
                    "scraped_timestamp": None,
                    "is_ground_truth": True,
                    "source": "ground_truth",
                    "ground_truth_query_ids": [query_id],
                    "ground_truth_query_texts": [query_text],
                    # snippets keyed by query_id — preserves per-query attribution
                    "snippets_by_query": {query_id: snippets},
                }
            else:
                # Same document referenced by an additional query — merge.
                if query_id not in gt_docs[citation]["ground_truth_query_ids"]:
                    gt_docs[citation]["ground_truth_query_ids"].append(query_id)
                    gt_docs[citation]["ground_truth_query_texts"].append(query_text)
                gt_docs[citation]["snippets_by_query"][query_id] = snippets

        queries.append({
            "query_id": query_id,
            "query_text": query_text,
            "user_answer": user_answer,
            "custom_instruction": custom_instruction,
            "batch_id": batch_id,
            "province": province,
            "ground_truth_citations": query_citations,
            # snippets for each GT document, scoped to this query
            "ground_truth_snippets": query_snippets,
        })

    return list(gt_docs.values()), queries


def _load_court_df(corpus_dir: str, court: str) -> pd.DataFrame | None:
    """Load one court's parquet file from the A2AJ canadian-case-law layout."""
    parquet_path = os.path.join(corpus_dir, court, "train.parquet")
    if not os.path.exists(parquet_path):
        return None
    return pd.read_parquet(parquet_path, engine="pyarrow")


def _normalize_corpus_row(row: dict) -> dict:
    """Map a canadian-case-law parquet row to the normalized schema."""
    citation = _str(row.get("citation_en"))
    court = _str(row.get("dataset")) or _parse_court_from_citation(citation)
    text = _str(row.get("unofficial_text_en"))
    return {
        "citation": citation,
        "citation2": _str(row.get("citation2_en")),
        "name": _str(row.get("name_en")),
        "court": court,
        "year": _parse_year_from_citation(citation),
        "char_count": len(text),
        "text": text,
        "url": _str(row.get("url_en")),
        "upstream_license": _str(row.get("upstream_license")),
        "document_date": row.get("document_date_en"),
        "scraped_timestamp": row.get("scraped_timestamp_en"),
        "is_ground_truth": False,
        "source": "canadian_case_law",
        "ground_truth_query_ids": [],
        "ground_truth_query_texts": [],
        "snippets_by_query": {},
    }


def _normalize_caseway_row(item: dict) -> dict:
    """Map a caseway JSON item to the normalized schema."""
    title = item.get("title") or ""
    match = _NEUTRAL_CITATION_RE.search(title)
    if match:
        citation = match.group(1)
        name = title[match.end():].strip()
    else:
        # Some caseway titles have the CanLII ID but the word "CanLII" was stripped,
        # e.g. "2012 48211 (NL SC) Western School District ...". Reconstruct the
        # canonical CanLII citation so deduplication works correctly.
        ck_match = _CANLII_NO_KEYWORD_RE.match(title)
        if ck_match:
            citation = f"{ck_match.group(1)} CanLII {ck_match.group(2)} ({ck_match.group(3)})"
            name = title[ck_match.end():].strip()
        else:
            citation = title or item.get("source_id") or ""
            name = title
    text = item.get("stitched_text") or ""
    return {
        "citation": citation,
        "citation2": "",
        "name": name,
        "court": (item.get("court") or "").upper(),
        "year": _parse_year_from_citation(citation),
        "char_count": len(text),
        "text": text,
        "url": item.get("url") or "",
        "upstream_license": "",
        "document_date": None,
        "scraped_timestamp": None,
        "is_ground_truth": False,
        "source": "caseway",
        "ground_truth_query_ids": [],
        "ground_truth_query_texts": [],
        "snippets_by_query": {},
    }


def _load_caseway(
    path: str, gt_citations: set[str], max_samples: int | None, seed: int
) -> list[dict]:
    """
    Load caseway stitched documents, skipping any whose citation already
    appears in the ground truth set. Optionally subsample to max_samples.
    """
    with open(path, "r", encoding="utf-8") as f:
        items = json.load(f)

    docs: list[dict] = []
    skipped = 0
    for item in items:
        doc = _normalize_caseway_row(item)
        if doc["citation"] in gt_citations:
            skipped += 1
            continue
        docs.append(doc)

    eligible = len(docs)
    if max_samples is not None and max_samples < eligible:
        docs = random.Random(seed).sample(docs, max_samples)
        print(f"  Loaded {len(docs)} caseway docs (sampled from {eligible} eligible; {skipped} skipped — GT overlap)")
    else:
        print(f"  Loaded {len(docs)} caseway docs ({skipped} skipped — GT overlap)")
    return docs


def _sample_corpus(
    corpus_dir: str,
    courts: list[str],
    excluded_citations: set[str],
    excluded_texts: dict[str, str],
    samples_per_court: int,
    seed: int,
    length_percentile_cutoff: int = 75,
    length_stats_sample: int = 20_000,
    debug: bool = False,
) -> list[dict]:
    """
    Stratified random sample from corpus courts, excluding GT/caseway citations.

    Parameters
    ----------
    corpus_dir         : root of the A2AJ per-court parquet folders
    courts             : list of court abbreviations to sample from
    excluded_citations : set of citations to exclude (GT + caseway)
    excluded_texts     : citation → text for excluded docs (used for collision comparison)
    samples_per_court  : target number of docs per court
    seed               : random seed for reproducibility
    length_percentile_cutoff : remove docs longer than this percentile (per court).
                               Computed from a random sample of length_stats_sample
                               docs.  Set to 100 to disable.
    length_stats_sample : number of docs to sample per court for length stats.
    """
    rng = random.Random(seed)
    sampled: list[dict] = []
    collisions_total = 0

    for court in courts:
        df = _load_court_df(corpus_dir, court)
        if df is None:
            print(f"  WARNING: Court '{court}' not found under {corpus_dir}. Skipping.")
            continue

        # Remove any document whose citation overlaps with excluded set
        collision_mask = df["citation_en"].isin(excluded_citations)
        if collision_mask.any():
            overlapping = df.loc[collision_mask, ["citation_en", "unofficial_text_en"]]
            collisions_total += len(overlapping)
            print(f"  [{court}] Skipped {len(overlapping)} doc(s) with overlapping citations:")
            for _, row in overlapping.iterrows():
                citation = row["citation_en"]
                if debug:
                    corpus_text = (row["unofficial_text_en"] or "")[:200].replace("\n", " ")
                    excluded_text = (excluded_texts.get(citation) or "")[:200].replace("\n", " ")
                    texts_match = (row["unofficial_text_en"] or "").strip() == (excluded_texts.get(citation) or "").strip()
                    print(f"    - {citation}  [texts {'MATCH' if texts_match else 'DIFFER'}]")
                    print(f"        corpus  : {corpus_text!r}")
                    print(f"        excluded: {excluded_text!r}")
                else:
                    print(f"    - {citation}")
        df = df[~collision_mask].reset_index(drop=True)

        # --- Length-based filtering (per court) ---
        if length_percentile_cutoff < 100:
            text_lengths = df["unofficial_text_en"].fillna("").str.len()

            stats_n = min(length_stats_sample, len(df))
            if stats_n < len(df):
                stats_idx = rng.sample(range(len(df)), stats_n)
                stats_lengths = text_lengths.iloc[stats_idx]
            else:
                stats_lengths = text_lengths

            cutoff = int(stats_lengths.quantile(length_percentile_cutoff / 100))
            before = len(df)
            df = df[text_lengths <= cutoff].reset_index(drop=True)
            removed = before - len(df)
            print(
                f"  [{court}] Length filter: p{length_percentile_cutoff} cutoff "
                f"= {cutoff:,d} chars, removed {removed:,d} doc(s) "
                f"({before:,d} → {len(df):,d})"
            )

        n = min(samples_per_court, len(df))
        if n < samples_per_court:
            print(
                f"  WARNING: [{court}] Only {len(df)} docs available after filtering; "
                f"sampling all {n}"
            )

        indices = rng.sample(range(len(df)), n)
        for idx in indices:
            row = df.iloc[idx].to_dict()
            row.setdefault("dataset", court)
            sampled.append(_normalize_corpus_row(row))

        print(f"  [{court}] Sampled {n} / {len(df)} available docs")

    if collisions_total:
        print(f"\n  Total GT-collision docs removed from corpus: {collisions_total}")

    return sampled


def _text_hash(text: str) -> str:
    """SHA-256 of whitespace-normalized, lowercased text. Used for text dedup."""
    normalized = " ".join((text or "").lower().split())
    return hashlib.sha256(normalized.encode()).hexdigest()


def _to_dataframe(docs: list[dict]) -> pd.DataFrame:
    """
    Convert list of normalized doc dicts to a pandas DataFrame.
    List- and dict-typed fields are JSON-serialized to strings for parquet compatibility.
    """
    # (field_name, fallback_if_missing_or_None)
    json_fields: list[tuple[str, object]] = [
        ("ground_truth_query_ids", []),
        ("ground_truth_query_texts", []),
        ("snippets_by_query", {}),
    ]
    rows = []
    for doc in docs:
        row = dict(doc)
        for field, default in json_fields:
            val = row.get(field)
            row[field] = json.dumps(val if val is not None else default)
        rows.append(row)
    return pd.DataFrame(rows)


def _find_near_duplicates_by_char_count(
    df: pd.DataFrame, threshold: float = 0.01
) -> list[dict]:
    """
    Return pairs of documents whose char_count is within `threshold` of each
    other (default 1%) AND whose year is the same.  Requiring the same year
    eliminates the bulk of coincidental length matches across unrelated cases.
    Documents with zero char_count or a null year are excluded.

    Uses a sort + sliding-window scan so the cost is O(n log n + n*k) where
    k is the average number of documents that fall inside a given window.
    """
    valid = (
        df[(df["char_count"] > 0) & df["year"].notna()]
        .sort_values("char_count")
        .reset_index(drop=True)
    )
    pairs: list[dict] = []
    n = len(valid)
    for i in range(n):
        ci = int(valid.at[i, "char_count"])
        yi = valid.at[i, "year"]
        for j in range(i + 1, n):
            cj = int(valid.at[j, "char_count"])
            if cj > ci * (1 + threshold):
                break
            if valid.at[j, "year"] != yi:
                continue
            diff_pct = round((cj - ci) / cj * 100, 3)
            pairs.append({
                "citation_a": valid.at[i, "citation"],
                "source_a": valid.at[i, "source"],
                "is_ground_truth_a": bool(valid.at[i, "is_ground_truth"]),
                "char_count_a": ci,
                "citation_b": valid.at[j, "citation"],
                "source_b": valid.at[j, "source"],
                "is_ground_truth_b": bool(valid.at[j, "is_ground_truth"]),
                "char_count_b": cj,
                "year": int(yi),
                "diff_pct": diff_pct,
            })
    return pairs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--ground_truth_path",
        default=DEFAULT_GROUND_TRUTH_PATH,
        help=(
            "Path to the human-annotated query export "
            f"(default: {DEFAULT_GROUND_TRUTH_PATH})"
        ),
    )
    parser.add_argument(
        "--corpus_dir",
        default=DEFAULT_CORPUS_DIR,
        help=(
            "Root of the A2AJ canadian-case-law per-court parquet folders "
            f"(default: {DEFAULT_CORPUS_DIR})"
        ),
    )
    parser.add_argument(
        "--courts",
        nargs="+",
        default=DEFAULT_COURTS,
        metavar="COURT",
        help=f"Courts to sample from (default: {DEFAULT_COURTS})",
    )

    sample_group = parser.add_mutually_exclusive_group()
    sample_group.add_argument(
        "--samples_per_court",
        type=int,
        default=DEFAULT_SAMPLES_PER_COURT,
        help=f"Fixed number of samples per court (default: {DEFAULT_SAMPLES_PER_COURT})",
    )
    sample_group.add_argument(
        "--samples_ratio",
        type=float,
        metavar="N",
        help=(
            "Sample N * (total GT doc count) docs per court "
            "(e.g. --samples_ratio 10 gives 10x GT docs per court). "
            "Mutually exclusive with --samples_per_court."
        ),
    )

    parser.add_argument(
        "--output_dir",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "test_dataset"),
        help="Output directory (default: outputs/test_dataset/)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--caseway_path",
        default=DEFAULT_CASEWAY_PATH,
        help=(
            "Path to the caseway JSON file. Pass an empty string to skip caseway docs. "
            f"(default: {DEFAULT_CASEWAY_PATH})"
        ),
    )
    parser.add_argument(
        "--caseway_samples",
        type=int,
        default=None,
        metavar="N",
        help="Max number of caseway docs to include (default: all)",
    )
    parser.add_argument(
        "--length_percentile_cutoff",
        type=int,
        default=75,
        metavar="P",
        help=(
            "Remove corpus (distractor) docs above this percentile of character "
            "length, computed per court from a sample of --length_stats_sample "
            "docs.  Set to 100 to disable (default: 75)."
        ),
    )
    parser.add_argument(
        "--length_stats_sample",
        type=int,
        default=20_000,
        metavar="N",
        help=(
            "Number of docs to sample per court for computing the length "
            "percentile cutoff (default: 20000)."
        ),
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Print text snippets for citation collisions to compare corpus vs excluded docs.",
    )
    parser.add_argument(
        "--no-dedup-text",
        dest="no_dedup_text",
        action="store_true",
        default=False,
        help=(
            "Skip deduplication by document text. By default, documents with "
            "identical text (after whitespace normalization) are deduplicated, "
            "keeping the first occurrence (GT takes priority)."
        ),
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Step 1: Load ground truth
    # ------------------------------------------------------------------
    print(f"[1/5] Loading ground truth from: {args.ground_truth_path}")
    gt_docs, queries = _load_ground_truth(args.ground_truth_path)
    gt_citations = {doc["citation"] for doc in gt_docs}
    print(
        f"      {len(gt_docs)} unique GT documents across {len(queries)} queries"
    )

    # Resolve sample count
    samples_per_court = args.samples_per_court
    if args.samples_ratio is not None:
        samples_per_court = max(1, int(args.samples_ratio * len(gt_docs)))
        print(
            f"      samples_ratio={args.samples_ratio}x → "
            f"{samples_per_court} samples per court"
        )

    # ------------------------------------------------------------------
    # Step 2: Load caseway docs
    # ------------------------------------------------------------------
    caseway_docs: list[dict] = []
    if args.caseway_path:
        print(f"\n[2/5] Loading caseway docs from: {args.caseway_path}")
        caseway_docs = _load_caseway(args.caseway_path, gt_citations, args.caseway_samples, args.seed)
    else:
        print("\n[2/5] Caseway docs: skipped (--caseway_path empty)")

    # Citations to exclude from corpus sampling (GT + caseway already covered)
    excluded_citations = gt_citations | {doc["citation"] for doc in caseway_docs}
    excluded_texts = {doc["citation"]: doc["text"] for doc in gt_docs + caseway_docs}

    # ------------------------------------------------------------------
    # Step 3: Sample corpus
    # ------------------------------------------------------------------
    print(f"\n[3/5] Sampling corpus from courts: {args.courts}")
    corpus_docs = _sample_corpus(
        args.corpus_dir, args.courts, excluded_citations, excluded_texts,
        samples_per_court, args.seed,
        length_percentile_cutoff=args.length_percentile_cutoff,
        length_stats_sample=args.length_stats_sample,
        debug=args.debug,
    )

    # ------------------------------------------------------------------
    # Step 4: Combine and deduplicate
    # ------------------------------------------------------------------
    print("\n[4/5] Combining and deduplicating …")
    all_docs = gt_docs + caseway_docs + corpus_docs
    df = _to_dataframe(all_docs)

    duplicates = df[df["citation"].duplicated(keep=False)]["citation"].tolist()
    if duplicates:
        print(
            f"  WARNING: {len(set(duplicates))} citations appear more than once. "
            "Keeping the first occurrence (ground truth takes priority since it is prepended)."
        )
        df = df.drop_duplicates(subset=["citation"], keep="first").reset_index(drop=True)

    # Text-based deduplication: catches same document cited under different strings.
    # Builds a remap so query ground-truth citations point to the surviving document.
    citation_remap: dict[str, str] = {}   # dropped citation → surviving citation
    citations_dropped: set[str] = set()   # dropped with no surviving equivalent

    if not args.no_dedup_text:
        empty_hash = _text_hash("")
        text_hashes = df["text"].apply(_text_hash)

        # Drop documents with empty text (no replacement available)
        empty_mask = text_hashes == empty_hash
        if empty_mask.any():
            print(f"  Text dedup: dropping {empty_mask.sum()} doc(s) with empty text:")
            for c, gt in zip(df.loc[empty_mask, "citation"], df.loc[empty_mask, "is_ground_truth"]):
                print(f"    - {c}{' [GT]' if gt else ''}")
                citations_dropped.add(c)
            df = df[~empty_mask].reset_index(drop=True)
            text_hashes = text_hashes[~empty_mask].reset_index(drop=True)

        # Log and drop duplicate texts (keep first — GT is prepended first so GT wins)
        dup_mask = text_hashes.duplicated(keep=False)
        if dup_mask.any():
            print(f"  Text dedup: found {text_hashes[dup_mask].nunique()} group(s) of identical texts:")
            for h in text_hashes[dup_mask].unique():
                group_citations = df.loc[text_hashes == h, "citation"].tolist()
                group_gt = df.loc[text_hashes == h, "is_ground_truth"].tolist()
                labels = [f"{c}{' [GT]' if g else ''}" for c, g in zip(group_citations, group_gt)]
                print(f"    {' | '.join(labels)}")
                surviving = group_citations[0]
                for dropped in group_citations[1:]:
                    citation_remap[dropped] = surviving
            df = df[~text_hashes.duplicated(keep="first")].reset_index(drop=True)
        else:
            print("  Text dedup: no duplicate document texts found.")
    else:
        print("  Text dedup: skipped (--no-dedup-text).")

    # Remap / remove ground-truth citations in queries to match surviving documents
    if citation_remap or citations_dropped:
        n_remapped = 0
        n_removed = 0
        for q in queries:
            new_citations: list[str] = []
            new_snippets: dict[str, list[str]] = {}
            for c in q["ground_truth_citations"]:
                if c in citations_dropped:
                    n_removed += 1
                    continue
                remapped = citation_remap.get(c, c)
                if remapped != c:
                    n_remapped += 1
                if remapped not in new_citations:
                    new_citations.append(remapped)
                snips = q.get("ground_truth_snippets", {}).get(c, [])
                if snips:
                    new_snippets.setdefault(remapped, []).extend(snips)
            q["ground_truth_citations"] = new_citations
            q["ground_truth_snippets"] = new_snippets
        if citation_remap:
            print(f"\n  Query citation remap: {n_remapped} reference(s) remapped across queries:")
            for dropped, surviving in citation_remap.items():
                print(f"    {dropped!r} → {surviving!r}")
        if citations_dropped:
            print(f"  Query citation removal: {n_removed} reference(s) to empty-text docs removed")

    # ------------------------------------------------------------------
    # Step 4b: Save unparseable court / year records for audit
    # ------------------------------------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)

    def _audit_records(subset: pd.DataFrame) -> list[dict]:
        records = []
        for _, row in subset.iterrows():
            records.append({
                "citation": row["citation"],
                "source": row["source"],
                "name": row["name"],
                "court": row["court"],
                "year": (int(row["year"]) if pd.notna(row.get("year")) else None),
                "url": row.get("url", ""),
                "is_ground_truth": bool(row["is_ground_truth"]),
                "query_ids": json.loads(row["ground_truth_query_ids"]) if row["ground_truth_query_ids"] else [],
                "query_texts": json.loads(row["ground_truth_query_texts"]) if row["ground_truth_query_texts"] else [],
            })
        return records

    unparseable_court = df[df["court"].fillna("") == ""]
    if not unparseable_court.empty:
        records = _audit_records(unparseable_court)
        unparseable_court_path = os.path.join(args.output_dir, "unparseable_court.json")
        with open(unparseable_court_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)
        print(f"\n  Saved {len(records)} doc(s) with unparseable court → {unparseable_court_path}")
    else:
        print("\n  No docs with unparseable court.")

    unparseable_year = df[df["year"].isna()]
    if not unparseable_year.empty:
        records = _audit_records(unparseable_year)
        unparseable_year_path = os.path.join(args.output_dir, "unparseable_year.json")
        with open(unparseable_year_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)
        print(f"  Saved {len(records)} doc(s) with unparseable year  → {unparseable_year_path}")
    else:
        print("  No docs with unparseable year.")

    # ------------------------------------------------------------------
    # Step 4c: Impute missing court and year values
    # ------------------------------------------------------------------
    imputation_log: dict = {"court_imputations": [], "year_imputations": None}

    # --- Court imputation: STATUTE vs UNKNOWN ---
    missing_court_mask = df["court"].fillna("") == ""
    if missing_court_mask.any():
        for idx in df.index[missing_court_mask]:
            citation = df.at[idx, "citation"]
            name = df.at[idx, "name"] or ""
            combined = f"{citation} {name}"
            label = "STATUTE" if _STATUTE_RE.search(combined) else "UNKNOWN"
            imputation_log["court_imputations"].append({
                "citation": citation,
                "original_court": "",
                "imputed_court": label,
            })
            df.at[idx, "court"] = label
        n_statute = sum(1 for r in imputation_log["court_imputations"] if r["imputed_court"] == "STATUTE")
        n_unknown = sum(1 for r in imputation_log["court_imputations"] if r["imputed_court"] == "UNKNOWN")
        print(f"\n  Court imputation: {n_statute} → STATUTE, {n_unknown} → UNKNOWN")

    # --- Year imputation: floor(mean) of known years ---
    missing_year_mask = df["year"].isna()
    if missing_year_mask.any():
        known_years = df.loc[~missing_year_mask, "year"]
        mean_year = int(known_years.mean())  # floor via int() truncation
        affected = df.loc[missing_year_mask, "citation"].tolist()
        df.loc[missing_year_mask, "year"] = mean_year
        imputation_log["year_imputations"] = {
            "imputed_value": mean_year,
            "computed_from": f"floor(mean of {len(known_years)} documents with parsed years)",
            "affected_citations": affected,
        }
        print(f"  Year imputation: {len(affected)} doc(s) → {mean_year} "
              f"(floor mean of {len(known_years)} known years)")

    imputation_path = os.path.join(args.output_dir, "imputations.json")
    with open(imputation_path, "w", encoding="utf-8") as f:
        json.dump(imputation_log, f, indent=2, ensure_ascii=False)
    print(f"  Saved imputation log → {imputation_path}")

    near_dup_pairs = _find_near_duplicates_by_char_count(df, threshold=0.01)
    near_dup_path = os.path.join(args.output_dir, "near_duplicate_char_counts.json")
    with open(near_dup_path, "w", encoding="utf-8") as f:
        json.dump(near_dup_pairs, f, indent=2, ensure_ascii=False)
    print(f"  Saved {len(near_dup_pairs)} near-duplicate pair(s) by char count → {near_dup_path}")

    # ------------------------------------------------------------------
    # Step 5: Write outputs
    # ------------------------------------------------------------------
    print(f"\n[5/5] Writing outputs to: {args.output_dir}")

    parquet_path = os.path.join(args.output_dir, "test_dataset.parquet")
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, parquet_path)
    print(f"  Wrote {len(df)} documents → {parquet_path}")

    json_path = os.path.join(args.output_dir, "test_dataset.json")
    datetime_cols = ["document_date", "scraped_timestamp"]
    json_df = df.copy()
    for col in datetime_cols:
        if col in json_df.columns:
            json_df[col] = json_df[col].apply(
                lambda v: v.isoformat() if pd.notna(v) else None
            )
    # Convert year float NaN → null so JSON output is valid
    if "year" in json_df.columns:
        json_df["year"] = json_df["year"].apply(
            lambda v: int(v) if pd.notna(v) else None
        )
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_df.to_dict(orient="records"), f, indent=2, ensure_ascii=False)
    print(f"  Wrote {len(df)} documents → {json_path}")

    queries_path = os.path.join(args.output_dir, "queries.json")
    with open(queries_path, "w", encoding="utf-8") as f:
        json.dump(queries, f, indent=2, ensure_ascii=False)
    print(f"  Wrote {len(queries)} queries    → {queries_path}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n── Summary ─────────────────────────────────────────")
    print(f"  Total documents    : {len(df)}")
    print(f"  Ground truth docs  : {int(df['is_ground_truth'].sum())}")
    print(f"  Caseway docs       : {int((df['source'] == 'caseway').sum())}")
    print(f"  Sampled corpus docs: {int((df['source'] == 'canadian_case_law').sum())}")
    per_court = df.groupby("court").size().to_dict()
    for court, count in sorted(per_court.items()):
        print(f"    {court:6s}: {count}")
    print(f"  Total queries      : {len(queries)}")
    print("────────────────────────────────────────────────────")


if __name__ == "__main__":
    main()