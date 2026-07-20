"""
Build a BM25 index from an existing index.chunks.pkl file.

The FAISS indexing pipeline (run_indexing.py) produces index.chunks.pkl but
only builds index.bm25.pkl when the retriever is a HybridRetriever or
BM25Retriever.  This script builds the .bm25.pkl from the chunks for any
existing index, so hybrid experiments can reuse a dense index.

Usage
-----
    python scripts/build_bm25_index.py --config configs/experiments/hybrid_bm25_qwen_kanon2rerank_recursive_4096.yaml

    # Or point directly at an index directory:
    python scripts/build_bm25_index.py --index-dir runs/indexes/qwen3_embedding_8b__recursive4096__d6fffb4580

    # Override BM25 parameters:
    python scripts/build_bm25_index.py --config ... --k1 1.2 --b 0.8
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmark_rag.components.retrievers.bm25_retriever import BM25Retriever


def main():
    parser = argparse.ArgumentParser(description="Build a BM25 index from index.chunks.pkl.")
    parser.add_argument("--config", default=None,
                        help="Experiment YAML — index dir is resolved from its index_id.")
    parser.add_argument("--index-dir", default=None,
                        help="Direct path to the index directory (overrides --config).")
    parser.add_argument("--k1", type=float, default=1.5, help="BM25 k1 parameter (default: 1.5)")
    parser.add_argument("--b", type=float, default=0.75, help="BM25 b parameter (default: 0.75)")
    parser.add_argument("--force", action="store_true",
                        help="Rebuild even if index.bm25.pkl already exists.")
    args = parser.parse_args()

    if args.index_dir:
        index_dir = Path(args.index_dir)
    elif args.config:
        from benchmark_rag.config.schemas import ExperimentConfig
        cfg = ExperimentConfig.from_yaml(args.config)
        index_dir = Path(cfg.indexing.output_dir)
    else:
        parser.error("Provide --config or --index-dir.")

    index_path = index_dir / "index"
    bm25_path = index_path.with_suffix(".bm25.pkl")
    chunks_path = index_path.with_suffix(".chunks.pkl")

    if not chunks_path.exists():
        sys.exit(f"ERROR: {chunks_path} not found. Run run_indexing.py first.")

    if bm25_path.exists() and not args.force:
        print(f"BM25 index already exists at {bm25_path}. Use --force to rebuild.")
        return

    print(f"Loading chunks from {chunks_path} ...")
    with open(chunks_path, "rb") as f:
        chunks = pickle.load(f)
    print(f"  {len(chunks)} chunks loaded.")

    print(f"Building BM25 index (k1={args.k1}, b={args.b}) ...")
    retriever = BM25Retriever(k1=args.k1, b=args.b)
    retriever.build_index(chunks)
    retriever.save_index(index_path)
    print(f"  Saved to {bm25_path}")


if __name__ == "__main__":
    main()
