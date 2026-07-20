# Input Data

Three inputs feed `build_test_dataset.py`. The first is private, so the build cannot be
re-run as-is — this folder documents the expected formats for transparency and replication
with other data.

## 1. Human-annotated queries (private) — `annotated_queries.json`

The export of the human annotation effort: a JSON list with one record per query.

| Field | Type | Description |
|---|---|---|
| `query_id` | int | Unique query ID |
| `query_text` | str | The query shown to the annotator |
| `province` | str | Canadian province/territory the query scenario is set in |
| `user_answer` | str | Reference answer written by the annotator (empty if annotation incomplete) |
| `custom_instruction` | str | Optional annotator instruction |
| `batch_id` | int | Annotation batch |
| `documents` | list | Documents the annotator linked to this query |
| `documents[].citation` | str | Citation as pasted by the annotator (normalized by the build script) |
| `documents[].title` | str | Case title |
| `documents[].source_url` | str | Where the annotator found the document |
| `documents[].full_text` | str | Full document text |
| `documents[].snippets` | list | Annotator-highlighted passages: `[{"text": ...}, ...]` |

## 2. Caseway corpus (private) — `filtered_caseway_data.json`

Same format as in [../../create_queries/data/README.md](../../create_queries/data/README.md):
a JSON list of records with `source_id`, `title`, `url`, `province`, `court`, and
`stitched_text`. Used as distractor documents.

## 3. A2AJ Canadian Legal Data (public) — `canadian-case-law/`

Same layout as in [../../create_queries/data/README.md](../../create_queries/data/README.md):
per-court parquet files at `canadian-case-law/{BCSC,CHRT,SST,TCC}/train.parquet`, pulled from
[A2AJ CLD](https://huggingface.co/datasets/a2aj/canadian-case-law). Used as distractor
documents via stratified sampling.