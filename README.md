# CanLegalRAGBench

Repository for the paper "CanLegalRAGBench: Evaluating Retrieval-Augmented Generation on
Canadian Case Law".

**Dataset:** [UBC-VL/CanLegalRAGBench on Hugging Face](https://huggingface.co/datasets/UBC-VL/CanLegalRAGBench)
## Repository layout

The repo is organized by benchmark stage. The two `create_*` folders document how the
dataset was built; the two `evaluate_*` folders are what you run to evaluate systems on the
released dataset.

| Folder | Purpose |
|---|---|
| [create_queries/](create_queries/) | Query creation pipeline: persona-conditioned generation from court decisions, LLM-judge filtering, and controlled query variations (the queries were then annotated by human legal experts) |
| [create_retrieval_dataset/](create_retrieval_dataset/) | How the retrieval corpus was assembled: annotated ground-truth documents + distractors, deduplication, and the released files (documentation of the as-run process) |
| [evaluate_retrieval/](evaluate_retrieval/) | Retrieval evaluation framework: chunking, embedding (Qwen / Gemma / Gemini / Kanon2), FAISS / BM25 / hybrid retrieval, reranking, IterRetGen; recall / precision / MRR / nDCG against ground-truth citations |
| [evaluate_generation/](evaluate_generation/) | End-to-end answer evaluation: generate answers from retrieval results, judge them against the human reference answers with Ragas (groundedness + factual precision) |

## Quick start (evaluating on the benchmark)

```bash
pip install -r requirements.txt

# 1. Download the dataset into evaluate_retrieval/data/ (see evaluate_retrieval/data/README.md)
huggingface-cli download UBC-VL/CanLegalRAGBench --repo-type dataset \
    --include "documents.parquet" "queries.json" --local-dir evaluate_retrieval/data/

# 2. Index and evaluate retrieval
cd evaluate_retrieval
python scripts/run_indexing.py  --config configs/experiments/qwen_recursive_8192.yaml
python scripts/run_benchmark.py --config configs/experiments/qwen_recursive_8192.yaml

# 3. Generate and judge answers
cd ../evaluate_generation
python 1_generate_answers.py --config ../evaluate_retrieval/configs/experiments/qwen_recursive_8192.yaml
python 2_prepare_eval_input.py --results ../evaluate_retrieval/runs/qwen_recursive_8192_1k-docs/results/query_results.jsonl
python 3_ragas_answer_eval.py --keep-answers
```

Each folder's README documents its stage in detail, including required API keys and data
formats.