"""
Append per-run cost summaries to a shared CSV log.

Indexing and evaluation each have their own default CSV — indexing is a one-off
per dataset/chunker/embedder, evaluation re-runs per experiment, and mixing
them makes trend analysis noisy.

CSV schema (same for both files):
    DATE_TIME, EXPERIMENT_ID, COST_OF_RUN, EMBEDDING_COST, RERANKER_COST,
    GENERATOR_COST, OTHER_COST, TOTAL_COST_SO_FAR

TOTAL_COST_SO_FAR is recomputed each call as the sum of COST_OF_RUN across all
prior rows plus the new row, so hand-editing the file remains consistent.
"""
from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path


_FIELDNAMES = [
    "DATE_TIME", "EXPERIMENT_ID", "COST_OF_RUN",
    "EMBEDDING_COST", "RERANKER_COST", "GENERATOR_COST", "OTHER_COST",
    "TOTAL_COST_SO_FAR",
]

DEFAULT_INDEXING_COST_CSV = Path("runs/cost_log_indexing.csv")
DEFAULT_BENCHMARK_COST_CSV = Path("runs/cost_log_benchmark.csv")

_SUBCOMPONENT_ATTRS = (
    "embedder", "retriever", "reranker", "generator",
    "intermediate_generator", "rewriter",
)


def _extract_cost(obj) -> float:
    """Read the tracked cost off one object; 0.0 when no cost is tracked."""
    if obj is None:
        return 0.0
    for attr in ("_total_est_cost_usd", "_total_cost"):
        v = getattr(obj, attr, None)
        if v is None:
            continue
        try:
            return float(v)
        except (TypeError, ValueError):
            continue
    return 0.0


def collect_component_costs(*roots) -> dict[str, float]:
    """
    Collect per-component cost breakdown from a pipeline and/or loose components.

    Returns a dict with keys: embedding, reranker, generator, other, total.
    """
    embedding_attrs = {"embedder"}
    reranker_attrs = {"reranker"}
    generator_attrs = {"generator", "intermediate_generator", "rewriter"}

    costs: dict[str, float] = {
        "embedding": 0.0,
        "reranker": 0.0,
        "generator": 0.0,
        "other": 0.0,
    }

    seen: set[int] = set()

    for root in roots:
        if root is None:
            continue
        # Root object's own cost (e.g. AgenticRAGPipeline._total_cost)
        root_cost = _extract_cost(root)
        if root_cost > 0 and id(root) not in seen:
            costs["other"] += root_cost
            seen.add(id(root))

        for attr in _SUBCOMPONENT_ATTRS:
            child = getattr(root, attr, None)
            if child is None or id(child) in seen:
                continue
            seen.add(id(child))
            child_cost = _extract_cost(child)
            if attr in embedding_attrs:
                costs["embedding"] += child_cost
            elif attr in reranker_attrs:
                costs["reranker"] += child_cost
            elif attr in generator_attrs:
                costs["generator"] += child_cost
            else:
                costs["other"] += child_cost

    costs["total"] = sum(costs.values())
    return costs


def sum_component_costs(*roots) -> float:
    """Sum tracked costs across a pipeline and/or loose components."""
    return collect_component_costs(*roots)["total"]


def append_cost_entry(
    csv_path: str | Path,
    experiment_id: str,
    cost_of_run_usd: float,
    cost_breakdown: dict[str, float] | None = None,
) -> tuple[float, Path]:
    """
    Append one row to the shared cost CSV, creating it with a header on first
    use.  Returns ``(running_total_usd, resolved_csv_path)``.
    """
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    prior_total = 0.0
    if csv_path.exists():
        with csv_path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    prior_total += float(row.get("COST_OF_RUN", 0.0) or 0.0)
                except ValueError:
                    pass

    cost = float(cost_of_run_usd)
    new_total = prior_total + cost
    write_header = not csv_path.exists()

    bd = cost_breakdown or {}

    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerow({
            "DATE_TIME": datetime.now().isoformat(timespec="seconds"),
            "EXPERIMENT_ID": experiment_id,
            "COST_OF_RUN": f"{cost:.6f}",
            "EMBEDDING_COST": f"{bd.get('embedding', 0.0):.6f}",
            "RERANKER_COST": f"{bd.get('reranker', 0.0):.6f}",
            "GENERATOR_COST": f"{bd.get('generator', 0.0):.6f}",
            "OTHER_COST": f"{bd.get('other', 0.0):.6f}",
            "TOTAL_COST_SO_FAR": f"{new_total:.6f}",
        })

    return new_total, csv_path
