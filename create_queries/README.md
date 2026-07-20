# Query Creation

This folder contains the pipeline that produced the candidate queries for CanLegalRAGBench.
Queries are generated from real Canadian court decisions, filtered by an LLM judge, expanded
into controlled variations, and then handed to human legal annotators (the annotation itself
is outside this repo — see the note at the bottom).

## Pipeline overview

```
court decisions (york parquet / caseway JSON)
        │
        ▼
1_generate_queries.py        persona-conditioned query generation (2 queries/doc:
        │                    layperson + legal associate, random Big Five traits,
        │                    random target section)
        ▼
2_filter_by_judge.py         LLM judge keeps/rejects each (query, document) pair;
        │                    readable output keeps layperson queries only
        ▼
3_create_query_variations.py each seed query → block of 8 variations
        │                    (2 fact variations × [base + 3 situation variations]),
        │                    each judged and lightly repaired; random province per block
        ▼
4_clean_and_group.py         normalize law-area labels; group blocks by law area
        │                    for efficient annotation
        ▼
annotation-ready query file (query_variations_cleaned.json)
```

## Setup

Install dependencies from the repository root (`pip install -r ../requirements.txt`) and put a
Gemini API key in a `.env` file in this folder (or export it):

```
GEMINI_API_KEY=...
```

Document data goes in `data/` — see [data/README.md](data/README.md) for the two sources
(the public A2AJ Canadian Legal Data courts, and the expected format for the private
provincial-court corpus).

All scripts write to `outputs/<experiment-name>/` under stable file names, so the steps chain
together directly. Run everything from this folder.

## Running the pipeline

```bash
# 1. Generate candidate queries from each document source
python 1_generate_queries.py -exp york_run -d york --num-samples 200
python 1_generate_queries.py -exp caseway_run -d caseway --caseway-file data/filtered_caseway_data.json

# 2. Judge each (query, document) pair
python 2_filter_by_judge.py -exp york_run -d york -in outputs/york_run/validation_set_york.json
python 2_filter_by_judge.py -exp caseway_run -d caseway -in outputs/caseway_run/validation_set_caseway.json

# 3. Expand seeds into variation blocks (both sources combined)
python 3_create_query_variations.py -exp query_variations \
    -in outputs/caseway_run/readable_judged_validation_set_caseway.csv \
        outputs/york_run/readable_judged_validation_set_york.csv

# 4. Clean and group for annotation
python 4_clean_and_group.py -in outputs/query_variations/query_variations.json
```

## The original run

In the run that produced the released benchmark, 200 york documents (SST, CHRT, TCC, BCSC;
TCC oversampled 2×) and 285 caseway documents were sampled; after filtering, 157 york and 41
caseway layperson seed queries remained, which step 3 expanded into 1,584 query variations
(198 blocks of 8). All generation, judging, and variation steps used
`gemini/gemini-2.5-flash`.

## Notes on reproducibility

- LLM outputs are not deterministic, so a rerun will not reproduce the released queries
  verbatim; the released benchmark queries are distributed with the dataset itself
  (see the repository README).
- The original run contained a small generation artifact in step 3 (the base row of each
  second fact-variation block recorded the seed query instead of the fact variation) that was
  repaired post-hoc. This release fixes the bug at the source, so step 4 no longer performs
  that repair.
- After step 4, the queries were annotated by human legal annotators, who searched for and
  highlighted the ground-truth documents and snippets for each query and wrote reference
  answers. The annotation tool and the aggregation of annotator output are not part of this
  repo; the resulting human-annotated retrieval dataset is published with the benchmark and
  consumed directly by `../create_retrieval_dataset/`.