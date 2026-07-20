#!/usr/bin/env python3
"""Step 3: Judge generated answers against the reference answers with Ragas.

For each record (from 2_prepare_eval_input.py), scores the generated answer's
groundedness in the human-written reference answer using Ragas
ResponseGroundedness (two-judge, 0-1 scale), and optionally factual precision
via Ragas FactualCorrectness (claim decomposition + per-claim NLI verdicts,
--also-factual-precision).

Outputs per-row scores (jsonl + csv) and an aggregate summary json. Use
--manual-eval to dump one row's decomposed claims, per-claim judgements, and
the exact judge prompts for human inspection, and --llm-log-jsonl to trace
every judge call.

Usage
-----
    python 3_ragas_answer_eval.py --input outputs/answers.jsonl --keep-answers

"""
from __future__ import annotations

import argparse
import asyncio
import contextvars
import csv
import json
import math
import os
import statistics
import sys
import threading
import zipfile
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tqdm import tqdm  # type: ignore[import-untyped]

DEFAULT_INPUT = "outputs/answers.jsonl"
DEFAULT_OUT_JSONL = "outputs/groundedness_scores.jsonl"
DEFAULT_OUT_CSV = "outputs/groundedness_scores.csv"
DEFAULT_SUMMARY_JSON = "outputs/groundedness_summary.json"

llm_call_context: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "llm_call_context",
    default=None,
)


def read_jsonl_from_zip(path: Path, member: str | None = None) -> list[dict[str, Any]]:
    with zipfile.ZipFile(path) as zf:
        if member is None:
            candidates = [name for name in zf.namelist() if name.endswith(".jsonl")]
            if not candidates:
                raise ValueError(f"No .jsonl file found inside {path}")
            if len(candidates) > 1:
                print(
                    f"Multiple .jsonl files found. Using {candidates[0]!r}. "
                    "Pass --zip-member to choose another one.",
                    file=sys.stderr,
                )
            member = candidates[0]

        with zf.open(member) as f:
            return [json.loads(line) for line in f if line.strip()]


def read_jsonl(path: Path, max_rows: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if max_rows is not None and len(rows) >= max_rows:
                break
    return rows


def load_records(
    path: Path, zip_member: str | None = None, max_rows: int | None = None
) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".zip":
        return read_jsonl_from_zip(path, member=zip_member)
    return read_jsonl(path, max_rows=max_rows)


def get_text(row: dict[str, Any], field: str) -> str:
    value = row.get(field, "")
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def score_value(result: Any) -> float | None:
    value = getattr(result, "value", result)
    if value is None:
        return None
    try:
        score = float(value)
    except TypeError:
        return None
    except ValueError:
        return None
    if not math.isfinite(score):
        return None
    return score


def is_finite_score(value: Any) -> bool:
    return score_value(value) is not None


def mean_or_none(values: Iterable[float | None]) -> float | None:
    clean = [score for value in values if (score := score_value(value)) is not None]
    if not clean:
        return None
    return statistics.fmean(clean)


class AsyncGenerateAdapter:
    """Expose async generation for Ragas metrics backed by a sync LLM client."""

    def __init__(self, llm: Any):
        self._llm = llm
        self.is_async = True

    def __getattr__(self, name: str) -> Any:
        return getattr(self._llm, name)

    def generate(self, prompt: str, response_model: type[Any]) -> Any:
        return self._llm.generate(prompt, response_model)

    async def agenerate(self, prompt: str, response_model: type[Any]) -> Any:
        return self._llm.generate(prompt, response_model)


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, str | int | bool):
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        return repr(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, set):
        return [json_safe(item) for item in sorted(value, key=repr)]
    if hasattr(value, "model_dump"):
        try:
            return json_safe(value.model_dump(mode="json"))
        except TypeError:
            return json_safe(value.model_dump())
    if hasattr(value, "dict"):
        try:
            return json_safe(value.dict())
        except TypeError:
            pass
    return repr(value)


def response_model_info(response_model: type[Any]) -> dict[str, str]:
    return {
        "name": getattr(response_model, "__name__", type(response_model).__name__),
        "module": getattr(response_model, "__module__", type(response_model).__module__),
    }


class LLMTraceLogger:
    def __init__(
        self,
        path: Path,
        *,
        provider: str,
        model: str,
        prompts_only: bool,
    ):
        self.path = Path(path)
        self.provider = provider
        self.model = model
        self.prompts_only = prompts_only
        self._call_index = 0
        self._thread_lock = threading.Lock()
        self._async_lock = asyncio.Lock()
        self._file = self.path.open("w", encoding="utf-8")
        self._closed = False

    @property
    def num_calls_logged(self) -> int:
        with self._thread_lock:
            return self._call_index

    def close(self) -> None:
        with self._thread_lock:
            if self._closed:
                return
            self._file.close()
            self._closed = True

    def log_success(
        self,
        *,
        prompt: str,
        response_model: type[Any],
        output: Any,
    ) -> None:
        self._write_record(
            self._build_record(prompt=prompt, response_model=response_model, output=output)
        )

    async def alog_success(
        self,
        *,
        prompt: str,
        response_model: type[Any],
        output: Any,
    ) -> None:
        async with self._async_lock:
            self.log_success(prompt=prompt, response_model=response_model, output=output)

    def log_error(
        self,
        *,
        prompt: str,
        response_model: type[Any],
        exc: Exception,
    ) -> None:
        self._write_record(
            self._build_record(prompt=prompt, response_model=response_model, exc=exc)
        )

    async def alog_error(
        self,
        *,
        prompt: str,
        response_model: type[Any],
        exc: Exception,
    ) -> None:
        async with self._async_lock:
            self.log_error(prompt=prompt, response_model=response_model, exc=exc)

    def _build_record(
        self,
        *,
        prompt: str,
        response_model: type[Any],
        output: Any | None = None,
        exc: Exception | None = None,
    ) -> dict[str, Any]:
        record: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "provider": self.provider,
            "model": self.model,
            "prompt": prompt,
            "response_model": response_model_info(response_model),
        }
        context = llm_call_context.get()
        if context:
            record.update(context)

        if exc is not None:
            record["error_type"] = type(exc).__name__
            record["error"] = str(exc)
            return record

        if not self.prompts_only:
            record["output"] = json_safe(output)
        return record

    def _write_record(self, record: dict[str, Any]) -> None:
        with self._thread_lock:
            if self._closed:
                raise RuntimeError("Cannot write to closed LLM trace logger")
            self._call_index += 1
            record = {"call_index": self._call_index, **record}
            self._file.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            self._file.flush()


class LLMLoggingAdapter:
    def __init__(self, llm: Any, logger: LLMTraceLogger):
        self._llm = llm
        self._logger = logger
        self.is_async = getattr(llm, "is_async", True)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._llm, name)

    def generate(self, prompt: str, response_model: type[Any]) -> Any:
        try:
            output = self._llm.generate(prompt, response_model)
        except Exception as exc:
            self._logger.log_error(prompt=prompt, response_model=response_model, exc=exc)
            raise
        self._logger.log_success(prompt=prompt, response_model=response_model, output=output)
        return output

    async def agenerate(self, prompt: str, response_model: type[Any]) -> Any:
        try:
            output = await self._llm.agenerate(prompt, response_model)
        except Exception as exc:
            await self._logger.alog_error(prompt=prompt, response_model=response_model, exc=exc)
            raise
        await self._logger.alog_success(
            prompt=prompt,
            response_model=response_model,
            output=output,
        )
        return output


def wrap_llm_with_logging(llm: Any, logger: LLMTraceLogger) -> Any:
    try:
        from ragas.llms.base import InstructorBaseRagasLLM
    except ImportError:
        return LLMLoggingAdapter(llm, logger)

    if isinstance(llm, InstructorBaseRagasLLM):
        class RagasLLMLoggingAdapter(LLMLoggingAdapter, InstructorBaseRagasLLM):
            pass

        return RagasLLMLoggingAdapter(llm, logger)

    return LLMLoggingAdapter(llm, logger)


def ensure_async_generation(llm: Any) -> Any:
    if getattr(llm, "is_async", True) is False and hasattr(llm, "generate"):
        try:
            from ragas.llms.base import InstructorBaseRagasLLM
        except ImportError:
            return AsyncGenerateAdapter(llm)

        if isinstance(llm, InstructorBaseRagasLLM):
            class RagasAsyncGenerateAdapter(AsyncGenerateAdapter, InstructorBaseRagasLLM):
                pass

            return RagasAsyncGenerateAdapter(llm)

        return AsyncGenerateAdapter(llm)
    return llm


def init_llm(provider: str, model: str, temperature: float):
    from dotenv import load_dotenv
    from ragas.llms import llm_factory

    load_dotenv()

    if provider == "google":
        api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Set GOOGLE_API_KEY or GEMINI_API_KEY before using --provider google"
            )
        from google import genai

        google_client = genai.Client(api_key=api_key)
        return ensure_async_generation(
            llm_factory(
                model,
                provider="google",
                client=google_client,
                temperature=temperature,
            )
        )

    if provider == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Set OPENAI_API_KEY before using --provider openai")
        from openai import OpenAI

        openai_client = OpenAI(api_key=api_key)
        return ensure_async_generation(
            llm_factory(
                model,
                provider="openai",
                client=openai_client,
                temperature=temperature,
            )
        )

    raise ValueError(f"Unsupported provider: {provider}")


def init_metrics(llm: Any, also_factual_precision: bool):
    try:
        from ragas.metrics.collections import FactualCorrectness, ResponseGroundedness
    except ImportError as exc:
        raise RuntimeError(
            "Could not import Ragas collection metrics. "
            "Install a recent version with: pip install 'ragas>=0.4'"
        ) from exc

    metrics: dict[str, Any] = {
        "groundedness": ResponseGroundedness(llm=llm),
    }

    if also_factual_precision:
        metrics["factual_precision"] = FactualCorrectness(
            llm=llm,
            mode="precision",
        )

    return metrics


def metric_context(
    row_index: int,
    row: dict[str, Any],
    args: argparse.Namespace,
    metric: str,
) -> dict[str, Any]:
    return {
        "row_index": row_index,
        "query_id": row.get("query_id"),
        "condition": row.get("condition"),
        "generator": row.get("generator"),
        "retrieval_method": row.get("retrieval_method"),
        "metric": metric,
        "provider": getattr(args, "provider", None),
        "model": getattr(args, "model", None),
    }


async def score_row(
    row_index: int,
    row: dict[str, Any],
    metrics: dict[str, Any],
    args: argparse.Namespace,
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    query = get_text(row, args.query_field)
    response = get_text(row, args.response_field)
    gold_answer = get_text(row, args.gold_field)

    out: dict[str, Any] = {
        "row_index": row_index,
        "query_id": row.get("query_id"),
        "condition": row.get("condition"),
        "generator": row.get("generator"),
        "retrieval_method": row.get("retrieval_method"),
        "query_text": query,
        "groundedness_gold": None,
        "groundedness_error": None,
    }

    if args.keep_answers:
        out["generated_answer"] = response
        out["ground_truth_answer"] = gold_answer

    if not response:
        out["groundedness_error"] = f"Missing response field {args.response_field!r}"
        return out

    if not gold_answer:
        out["groundedness_error"] = f"Missing gold field {args.gold_field!r}"
        return out

    async with semaphore:
        try:
            token = llm_call_context.set(
                metric_context(row_index, row, args, metric="groundedness")
            )
            try:
                result = await metrics["groundedness"].ascore(
                    response=response,
                    retrieved_contexts=[gold_answer],
                )
            finally:
                llm_call_context.reset(token)
            score = score_value(result)
            if score is None:
                out["groundedness_error"] = "Groundedness metric returned a non-finite score"
            else:
                out["groundedness_gold"] = score
        except Exception as exc:
            out["groundedness_error"] = repr(exc)

        if "factual_precision" in metrics:
            out["factual_precision_gold"] = None
            out["factual_precision_error"] = None
            try:
                token = llm_call_context.set(
                    metric_context(row_index, row, args, metric="factual_precision")
                )
                try:
                    result = await metrics["factual_precision"].ascore(
                        response=response,
                        reference=gold_answer,
                    )
                finally:
                    llm_call_context.reset(token)
                score = score_value(result)
                if score is None:
                    out["factual_precision_error"] = (
                        "Factual precision metric returned a non-finite score"
                    )
                else:
                    out["factual_precision_gold"] = score
            except Exception as exc:
                out["factual_precision_error"] = repr(exc)

    return out


async def manual_eval_row(
    row_index: int,
    row: dict[str, Any],
    metrics: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Run the judge on a single row and capture every intermediate artifact.

    Unlike ``score_row`` (which only keeps the final scalar), this reproduces the
    FactualCorrectness precision chain step-by-step so a human can eyeball whether
    the judge is reasonable:
      1. decompose the generated answer into standalone claims (statements)
      2. NLI-judge each claim against the gold answer (verdict 0/1 + reason)
      3. compute precision = supported / total
    It also captures the ResponseGroundedness dual-judge ratings and, crucially,
    the exact prompt strings that were sent to the model.
    """
    from ragas.metrics.collections.factual_correctness.metric import (
        ClaimDecompositionInput,
        NLIStatementInput,
    )
    from ragas.metrics.collections.response_groundedness.metric import (
        ResponseGroundednessInput,
    )

    query = get_text(row, args.query_field)
    response = get_text(row, args.response_field)
    gold_answer = get_text(row, args.gold_field)

    out: dict[str, Any] = {
        "row_index": row_index,
        "query_id": row.get("query_id"),
        "condition": row.get("condition"),
        "generator": row.get("generator"),
        "retrieval_method": row.get("retrieval_method"),
        "query_text": query,
        "generated_answer": response,
        "ground_truth_answer": gold_answer,
        "provider": args.provider,
        "model": args.model,
        "temperature": args.temperature,
    }

    if not response:
        out["error"] = f"Missing response field {args.response_field!r}"
        return out
    if not gold_answer:
        out["error"] = f"Missing gold field {args.gold_field!r}"
        return out

    # --- FactualCorrectness precision: decompose -> judge each claim ---
    fc = metrics["factual_precision"]
    fc_block: dict[str, Any] = {
        "mode": fc.mode,
        "atomicity": fc.atomicity,
        "coverage": fc.coverage,
    }

    decomp_input = ClaimDecompositionInput(
        response=response, atomicity=fc.atomicity, coverage=fc.coverage
    )
    fc_block["decomposition_prompt"] = fc.prompt.to_string(decomp_input)

    token = llm_call_context.set(
        metric_context(row_index, row, args, metric="claim_decomposition")
    )
    try:
        claims = await fc._decompose_claims(response)
    finally:
        llm_call_context.reset(token)
    fc_block["decomposed_statements"] = list(claims)
    fc_block["num_statements"] = len(claims)

    if claims:
        nli_input = NLIStatementInput(context=gold_answer, statements=list(claims))
        fc_block["nli_prompt"] = fc.nli_prompt.to_string(nli_input)

        token = llm_call_context.set(
            metric_context(row_index, row, args, metric="nli_verification")
        )
        try:
            verdicts = await fc._verify_claims(list(claims), gold_answer)
        finally:
            llm_call_context.reset(token)

        judgements = [
            {
                "statement": stmt.statement,
                "verdict": int(stmt.verdict),
                "supported": bool(stmt.verdict),
                "reason": stmt.reason,
            }
            for stmt in verdicts.statements
        ]
        fc_block["judgements"] = judgements
        tp = sum(1 for j in judgements if j["verdict"] == 1)
        fp = sum(1 for j in judgements if j["verdict"] != 1)
        fc_block["true_positives"] = tp
        fc_block["false_positives"] = fp
        fc_block["precision"] = tp / (tp + fp + 1e-8) if (tp + fp) else None
    else:
        fc_block["judgements"] = []
        fc_block["precision"] = None

    out["factual_precision"] = fc_block

    # --- ResponseGroundedness: dual-judge whole-answer ratings (for reference) ---
    grd = metrics["groundedness"]
    grd_input = ResponseGroundednessInput(response=response, context=gold_answer)
    token = llm_call_context.set(
        metric_context(row_index, row, args, metric="groundedness")
    )
    try:
        judge1 = await grd._get_judge_rating(grd.judge1_prompt, response, gold_answer)
        judge2 = await grd._get_judge_rating(grd.judge2_prompt, response, gold_answer)
    finally:
        llm_call_context.reset(token)
    grd_score = grd._average_scores(judge1 / 2.0, judge2 / 2.0)
    out["groundedness"] = {
        "judge1_prompt": grd.judge1_prompt.to_string(grd_input),
        "judge2_prompt": grd.judge2_prompt.to_string(grd_input),
        "judge1_rating_0_2": None if math.isnan(judge1) else judge1,
        "judge2_rating_0_2": None if math.isnan(judge2) else judge2,
        "score_0_1": None if math.isnan(grd_score) else grd_score,
        "note": "0=not grounded, 1=partially, 2=fully; score is mean of the two judges on a 0-1 scale.",
    }

    return out


def render_manual_eval_md(out: dict[str, Any]) -> str:
    """Render the manual-eval artifact as human-readable Markdown."""
    lines: list[str] = []
    lines.append(f"# Judge manual-eval — row {out.get('row_index')} "
                 f"(query_id={out.get('query_id')})")
    lines.append("")
    lines.append(f"- **provider/model**: {out.get('provider')} / {out.get('model')} "
                 f"(temperature={out.get('temperature')})")
    lines.append(f"- **condition**: {out.get('condition')}  "
                 f"**generator**: {out.get('generator')}  "
                 f"**retrieval_method**: {out.get('retrieval_method')}")
    lines.append("")
    if out.get("error"):
        lines.append(f"> ERROR: {out['error']}")
        return "\n".join(lines)

    lines.append("## Query")
    lines.append("")
    lines.append(out.get("query_text", ""))
    lines.append("")
    lines.append("## Generated answer (what the judge decomposes)")
    lines.append("")
    lines.append(out.get("generated_answer", ""))
    lines.append("")
    lines.append("## Ground-truth answer (the context claims are judged against)")
    lines.append("")
    lines.append(out.get("ground_truth_answer", ""))
    lines.append("")

    fc = out.get("factual_precision", {})
    lines.append("## FactualCorrectness (precision) — decomposed statements + judgements")
    lines.append("")
    lines.append(f"- atomicity={fc.get('atomicity')}, coverage={fc.get('coverage')}")
    lines.append(f"- **precision** = {fc.get('precision')}  "
                 f"(TP={fc.get('true_positives')}, FP={fc.get('false_positives')}, "
                 f"n={fc.get('num_statements')})")
    lines.append("")
    for i, j in enumerate(fc.get("judgements", []), start=1):
        mark = "✅ supported (1)" if j["verdict"] == 1 else "❌ unsupported (0)"
        lines.append(f"### Statement {i} — {mark}")
        lines.append("")
        lines.append(f"> {j['statement']}")
        lines.append("")
        lines.append(f"**Judge reason:** {j['reason']}")
        lines.append("")

    grd = out.get("groundedness", {})
    lines.append("## ResponseGroundedness (whole-answer dual judge, for reference)")
    lines.append("")
    lines.append(f"- judge1 rating (0-2): {grd.get('judge1_rating_0_2')}")
    lines.append(f"- judge2 rating (0-2): {grd.get('judge2_rating_0_2')}")
    lines.append(f"- **score (0-1)**: {grd.get('score_0_1')}")
    lines.append(f"- {grd.get('note', '')}")
    lines.append("")

    lines.append("---")
    lines.append("## Exact prompts sent to the judge")
    lines.append("")
    lines.append("<details><summary>Claim decomposition prompt</summary>")
    lines.append("")
    lines.append("```")
    lines.append(fc.get("decomposition_prompt", ""))
    lines.append("```")
    lines.append("</details>")
    lines.append("")
    lines.append("<details><summary>NLI verification prompt</summary>")
    lines.append("")
    lines.append("```")
    lines.append(fc.get("nli_prompt", ""))
    lines.append("```")
    lines.append("</details>")
    lines.append("")
    lines.append("<details><summary>Groundedness judge 1 prompt</summary>")
    lines.append("")
    lines.append("```")
    lines.append(grd.get("judge1_prompt", ""))
    lines.append("```")
    lines.append("</details>")
    lines.append("")
    lines.append("<details><summary>Groundedness judge 2 prompt</summary>")
    lines.append("")
    lines.append("```")
    lines.append(grd.get("judge2_prompt", ""))
    lines.append("```")
    lines.append("</details>")
    lines.append("")
    return "\n".join(lines)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return

    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groundedness_values = [row.get("groundedness_gold") for row in rows]
    failed = [
        row
        for row in rows
        if row.get("groundedness_error") or not is_finite_score(row.get("groundedness_gold"))
    ]

    summary: dict[str, Any] = {
        "num_rows": len(rows),
        "num_groundedness_scored": len(rows) - len(failed),
        "num_groundedness_failed": len(failed),
        "mean_groundedness_gold": mean_or_none(groundedness_values),
    }

    if rows and "factual_precision_gold" in rows[0]:
        factual_values = [row.get("factual_precision_gold") for row in rows]
        factual_failed = [
            row
            for row in rows
            if row.get("factual_precision_error")
            or not is_finite_score(row.get("factual_precision_gold"))
        ]
        summary.update(
            {
                "num_factual_precision_scored": len(rows) - len(factual_failed),
                "num_factual_precision_failed": len(factual_failed),
                "mean_factual_precision_gold": mean_or_none(factual_values),
            }
        )

    return summary


async def run(args: argparse.Namespace) -> None:
    input_path = Path(args.input)
    out_jsonl = Path(args.out_jsonl)
    out_csv = Path(args.out_csv)
    summary_json = Path(args.summary_json)

    # For manual eval we only ever touch the first row after --start, so avoid
    # parsing the entire (potentially multi-hundred-MB) input file.
    max_rows = (args.start + 1) if args.manual_eval else None
    records = load_records(input_path, zip_member=args.zip_member, max_rows=max_rows)

    if args.start:
        records = records[args.start :]
    if args.limit is not None:
        records = records[: args.limit]

    print(f"Loaded {len(records)} rows from {input_path}", file=sys.stderr)
    print(f"Initializing evaluator: provider={args.provider}, model={args.model}", file=sys.stderr)

    llm_logger: LLMTraceLogger | None = None
    llm = init_llm(
        provider=args.provider,
        model=args.model,
        temperature=args.temperature,
    )
    llm_log_jsonl = getattr(args, "llm_log_jsonl", None)
    if llm_log_jsonl:
        llm_logger = LLMTraceLogger(
            Path(llm_log_jsonl),
            provider=args.provider,
            model=args.model,
            prompts_only=getattr(args, "llm_log_prompts_only", False),
        )
        llm = wrap_llm_with_logging(llm, llm_logger)

    metrics = init_metrics(
        llm=llm,
        # manual eval always needs the decomposition metric
        also_factual_precision=args.also_factual_precision or args.manual_eval,
    )

    if args.manual_eval:
        try:
            if not records:
                raise ValueError("No records to evaluate (check --start/--limit/--input)")
            row_index = args.start
            row = records[0]
            print(
                f"Manual-eval on 1 row (row_index={row_index}, "
                f"query_id={row.get('query_id')})",
                file=sys.stderr,
            )
            out = await manual_eval_row(row_index, row, metrics, args)
        finally:
            if llm_logger is not None:
                llm_logger.close()

        out_json = Path(args.manual_eval_json)
        out_md = Path(args.manual_eval_md)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_md.parent.mkdir(parents=True, exist_ok=True)
        with out_json.open("w", encoding="utf-8") as f:
            json.dump(json_safe(out), f, indent=2, ensure_ascii=False, allow_nan=False)
        out_md.write_text(render_manual_eval_md(out), encoding="utf-8")
        print(f"Wrote {out_json}", file=sys.stderr)
        print(f"Wrote {out_md}", file=sys.stderr)
        return

    scored: list[dict[str, Any]] = []
    try:
        semaphore = asyncio.Semaphore(args.concurrency)
        tasks = [
            asyncio.create_task(score_row(i, row, metrics, args, semaphore))
            for i, row in enumerate(records)
        ]

        with tqdm(
            total=len(tasks),
            desc="Scoring rows",
            unit="row",
            file=sys.stderr,
            miniters=args.progress_every,
        ) as progress:
            for task in asyncio.as_completed(tasks):
                scored.append(await task)
                progress.update(1)
    finally:
        if llm_logger is not None:
            llm_logger.close()

    scored.sort(key=lambda row: row["row_index"])
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_jsonl, scored)
    write_csv(out_csv, scored)

    summary = summarize(scored)
    if llm_logger is not None:
        summary.update(
            {
                "llm_log_jsonl": str(llm_logger.path),
                "num_llm_calls_logged": llm_logger.num_calls_logged,
            }
        )
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, allow_nan=False)

    print(json.dumps(summary, indent=2), file=sys.stderr)
    print(f"Wrote {out_jsonl}", file=sys.stderr)
    print(f"Wrote {out_csv}", file=sys.stderr)
    print(f"Wrote {summary_json}", file=sys.stderr)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate Ragas groundedness of generated answers in gold answers."
    )
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--zip-member", default=None)
    parser.add_argument("--out-jsonl", default=DEFAULT_OUT_JSONL)
    parser.add_argument("--out-csv", default=DEFAULT_OUT_CSV)
    parser.add_argument("--summary-json", default=DEFAULT_SUMMARY_JSON)
    parser.add_argument("--llm-log-jsonl", default=None)
    parser.add_argument("--llm-log-prompts-only", action="store_true")

    parser.add_argument("--query-field", default="query_text")
    parser.add_argument("--response-field", default="generated_answer")
    parser.add_argument("--gold-field", default="ground_truth_answer")

    parser.add_argument("--provider", choices=["google", "openai"], default="google")
    parser.add_argument("--model", default="gemini-2.5-pro")
    parser.add_argument("--temperature", type=float, default=0.0)

    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=1)
    parser.add_argument("--keep-answers", action="store_true")
    parser.add_argument("--also-factual-precision", action="store_true")

    parser.add_argument(
        "--manual-eval",
        action="store_true",
        help="Run the judge on a single row and dump decomposed statements + "
        "per-statement judgements + exact prompts for manual review.",
    )
    parser.add_argument(
        "--manual-eval-json",
        default="outputs/manual_eval/judge_manual_eval.json",
    )
    parser.add_argument(
        "--manual-eval-md",
        default="outputs/manual_eval/judge_manual_eval.md",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.concurrency < 1:
        raise ValueError("--concurrency must be at least 1")
    if args.progress_every < 1:
        raise ValueError("--progress-every must be at least 1")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
