# Benchmark Data

Download the released dataset from
[UBC-VL/CanLegalRAGBench](https://huggingface.co/datasets/UBC-VL/CanLegalRAGBench) and place
the files here:

```
data/
├── documents.parquet   # 1,117 documents (corpus to index)
└── queries.json        # 532 queries (json-lines)
```

For example:

```bash
huggingface-cli download UBC-VL/CanLegalRAGBench --repo-type dataset \
    --include "documents.parquet" "queries.json" --local-dir data/
```

## Schemas

`documents.parquet` — one row per document:

| Field | Description |
|---|---|
| `citation` | Canonical citation, used as the document ID for retrieval matching |
| `name` | Case name / style of cause |
| `original_source` | Court or body abbreviation (`STATUTE` / `UNKNOWN` when unparseable) |
| `year` | Decision year |
| `text` | Full document text |
| `url` | Source URL |
| `upstream_license` | Upstream license, or an unofficial-reproduction notice |
| `is_ground_truth` | Whether an annotator linked this document to at least one query |
| `dataset_source` | `annotator`, `caseway`, or `canadian_case_law` |
| `ground_truth_query_ids` | Query IDs this document answers |

`queries.json` — one json-lines record per query:

| Field | Description |
|---|---|
| `query_id` | Unique query ID |
| `query_text` | The query (the benchmark prepends "I am in {province}." at retrieval time) |
| `answer` | Reference answer written by a human annotator |
| `batch_id` | Annotation batch |
| `ground_truth_citations` | Citations of the documents that answer this query |
| `province` | Province/territory the query scenario is set in |