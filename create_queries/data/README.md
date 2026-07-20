# Document Data

Two document corpora feed the query-generation pipeline.

## 1. York / A2AJ Canadian Legal Data (public)

Pull the court subsets from [A2AJ CLD](https://huggingface.co/datasets/a2aj/canadian-case-law)
and place each court's parquet file at:

```
data/canadian-case-law/
├── SST/train.parquet     # Social Security Tribunal
├── CHRT/train.parquet    # Canadian Human Rights Tribunal
├── TCC/train.parquet     # Tax Court of Canada
└── BCSC/train.parquet    # Supreme Court of British Columbia
```

The pipeline uses these columns: `dataset` (court code), `citation_en`, and
`unofficial_text_en` (full English decision text).

## 2. Caseway provincial-court corpus (private)

The second corpus is a collection of provincial / small-claims / family court decisions that
was obtained privately and cannot be redistributed. To replicate this part of the pipeline,
substitute any comparable collection of decisions from lower Canadian courts, formatted as a
single JSON file (`data/filtered_caseway_data.json`): a list of records with the fields

| Field | Type | Description |
|---|---|---|
| `source_id` | str | Unique document identifier |
| `title` | str | Case title / style of cause |
| `url` | str | Source URL of the decision |
| `language` | str | Document language (the pipeline assumes English) |
| `province` | str | Province or territory of the court |
| `court` | str | Lowercase court code (e.g. `oncj`, `bcpc`, `onscsm`) |
| `chunk_count` | int | Number of source chunks the text was stitched from (informational) |
| `stitched_text` | str | Full decision text |

Before query generation, the corpus was pre-filtered to documents whose word counts fall
between the corpus's first and third length quartiles (the same filter
`1_generate_queries.py` applies to the York data automatically).