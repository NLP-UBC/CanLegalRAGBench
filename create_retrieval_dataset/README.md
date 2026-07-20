# Retrieval Dataset Creation

This folder documents how the CanLegalRAGBench retrieval corpus was assembled. **You do not
need to run anything here to use the benchmark** — the finished dataset is released at
[UBC-VL/CanLegalRAGBench](https://huggingface.co/datasets/UBC-VL/CanLegalRAGBench), and the
evaluation folders (`../evaluate_retrieval/`, `../evaluate_generation/`) consume it directly.
The code is provided so the construction process is transparent and can be replicated with
other data.

## How the corpus was built

The queries from `../create_queries/` were annotated by human legal annotators, who searched
for the court decisions answering each query, highlighted the relevant snippets, and wrote a
reference answer. see [data/README.md](data/README.md) for its schema.

[build_test_dataset.py](build_test_dataset.py) then combines three document sources into one
deduplicated corpus:

1. **Ground-truth documents** — every document an annotator linked to a query.
2. **Caseway distractors** — provincial/lower-court decisions from a privately obtained
   corpus (same source as in `../create_queries/`).
3. **Corpus distractors** — a stratified random sample from the public A2AJ Canadian Legal
   Data courts (BCSC, CHRT, SST, TCC), with documents above the 75th length percentile
   removed (seed 42).

The script deduplicates by citation and by normalized text (ground truth wins; query
citation lists are remapped to the surviving document), labels documents whose citation
parses to no known court as `STATUTE` or `UNKNOWN`, imputes missing years, and writes audit
logs for every one of those decisions alongside the outputs:

```
outputs/test_dataset/
├── test_dataset.parquet / test_dataset.json   # the document corpus
├── queries.json                                # queries with ground-truth citations
├── unparseable_court.json / unparseable_year.json
├── imputations.json
└── near_duplicate_char_counts.json
```

## From the built dataset to the released dataset

The released Hugging Face dataset was derived from these outputs by:

1. Dropping queries whose annotation was never completed (no reference answer).
2. Dropping all Quebec-province queries (Quebec's civil-law system was deemed out of scope
   partway through the project) and a small number of queries manually screened as not well
   answered.
3. Renaming columns for public consumption (`user_answer` → `answer`, `court` →
   `original_source`, `source` → `dataset_source` with `ground_truth` → `annotator`),
   dropping internal columns (annotation snippets, char counts, timestamps), normalizing
   court-code variants, and filling missing `upstream_license` values with an
   unofficial-reproduction notice.
