# Retrieval Evaluation

A modular framework for evaluating retrieval pipelines on the CanLegalRAGBench dataset.
Documents are chunked, embedded, and indexed once; retrieval experiments then reuse the index
and report recall / precision / MRR / nDCG / hit@k against the human-annotated ground-truth
citations.

## Architecture

**Indexing is separated from evaluation** so experiments sharing the same dataset + chunker +
embedder reuse a single index:

```
Stage 1 (run once): documents.parquet → Chunker → Embedder → FAISS index (on disk)
Stage 2 (run many): query → Embedder → Retrieval → [Reranker] → metrics
```

Three pipeline types share this structure:

| Pipeline | Retrieval | File |
|---|---|---|
| `RAGPipeline` | dense (FAISS) top-k, optional reranker | `benchmark_rag/pipeline/rag_pipeline.py` |
| `HybridRAGPipeline` | FAISS + BM25 union → Kanon2 cross-encoder rerank (RRF fallback) | `benchmark_rag/pipeline/hybrid_pipeline.py` |
| `IterRetGenPipeline` | iterative retrieve → generate → re-retrieve | `benchmark_rag/pipeline/iterretgen_pipeline.py` |

plus a chunk- or document-level **BM25** baseline (`scripts/run_bm25_benchmark.py`).

All components (chunkers, embedders, retrievers, rerankers, generators) are referenced by
dotted type paths in YAML and instantiated via `benchmark_rag/registry.py` — adding a new
embedder is one new file plus a YAML reference, with no pipeline changes. Every experiment
YAML inherits from `configs/base.yaml` via deep merge. The index directory name is a
deterministic hash of (dataset, chunker, embedder), so experiments that differ only in
k-values or reranker share an index automatically.

Available components:

- **Chunkers**: recursive (primary), naive, semantic
- **Embedders**: Qwen3-Embedding (local), EmbeddingGemma (local), Gemini (API), Kanon2 (legal-domain API)
- **Retrievers**: FAISS (dense), BM25 (`L` or `okapi`), hybrid
- **Reranker**: Kanon2 cross-encoder (API)
- **Generators** (for IterRetGen / optional answer generation): Gemini, Qwen, Gemma

## Setup

```bash
pip install -r ../requirements.txt
```

Put the benchmark data in `data/` (see [data/README.md](data/README.md)), and set API keys in
a `.env` file in this folder as needed:

| Variable | Required for |
|---|---|
| `GOOGLE_API_KEY` (or `GEMINI_API_KEY`) | Gemini embedder / generator / judge |
| `ISAACUS_API_KEY` | Kanon2 embedder / reranker |
| `HF_TOKEN` | EmbeddingGemma (gated model) |

GPU is recommended for the Qwen embedder (8B parameters); Gemma (300M) runs on modest GPUs.

## Running experiments

```bash
# 1. Build the index (once per dataset + chunker + embedder combination)
python scripts/run_indexing.py --config configs/experiments/qwen_recursive_8192.yaml

# 2. Evaluate retrieval
python scripts/run_benchmark.py --config configs/experiments/qwen_recursive_8192.yaml

# Reranked variant (reuses the same index)
python scripts/run_benchmark.py --config configs/experiments/gemma_recursive_8192_rerank.yaml

# IterRetGen (iterative retrieval-generation)
python scripts/run_benchmark.py --config configs/experiments/qwen_recursive_8192_iterretgen_rerank.yaml --iterretgen

# BM25 baseline (chunk-level; add --doc-level to index whole documents)
python scripts/run_bm25_benchmark.py --config configs/experiments/bm25_okapi_recursive_4096.yaml

# Hybrid FAISS+BM25 (build the BM25 side of the index first)
python scripts/build_bm25_index.py --config configs/experiments/hybrid_bm25_qwen_kanon2rerank_recursive_4096.yaml
python scripts/run_benchmark.py --config configs/experiments/hybrid_bm25_qwen_kanon2rerank_recursive_4096.yaml
```

Useful flags for `run_benchmark.py`: `--resume` (skip queries already in
`query_results.jsonl`), `--generate` (also generate answers with the configured generator),
`--judge` (score generated answers against the reference with an LLM judge).

Every query is prefixed with `"I am in {province}."` before retrieval, using the query's
`province` field.

## Configs

`configs/experiments/` contains one representative config per pipeline family. To sweep other
settings, copy a config and change the relevant keys — chunk size (`chunker.max_chunk_chars`),
embedder, `evaluation.k_values`, or add a `reranker:` block. Reranked experiments evaluate at
k ≤ 25.

The paper's experiments used `dataset.max_docs: 1000` (the first 1,000 documents of the
corpus, which places all ground-truth documents plus distractors in the index); set it to
`null` to index all 1,117 released documents.

## Outputs

```
runs/
├── indexes/{index_id}/            # shared FAISS/BM25 indexes
└── {experiment_id}/
    ├── config.json                # resolved config snapshot
    ├── logs/
    └── results/
        ├── metrics.json           # aggregate scores by metric and k
        └── query_results.jsonl    # per-query retrieved IDs and scores
```

API costs are appended per run to `runs/cost_log_indexing.csv` and
`runs/cost_log_benchmark.csv`.