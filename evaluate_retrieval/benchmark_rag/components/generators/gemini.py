"""
Gemini 2.5 Flash generator and Gemini 2.5 Pro judge.

Uses the google-genai SDK (google.genai), the successor to google-generativeai.
"""
from __future__ import annotations

import logging
import time

from benchmark_rag.components.base import BaseGenerator, RetrievedChunk
from benchmark_rag.prompts.answer_generator import ANSWER_SYSTEM_PROMPT
from benchmark_rag.prompts.judge import JUDGE_SYSTEM_PROMPT

log = logging.getLogger(__name__)

_RETRY_DELAYS = [5, 10, 30, 60, 120]  # initial backoff; after exhausted, repeats 120s forever


def _generate_with_retry(client, **kwargs):
    """Call models.generate_content with infinite retry on 429/503."""
    from google.genai.errors import ClientError, ServerError

    attempt = 0
    while True:
        try:
            return client.models.generate_content(**kwargs)
        except (ClientError, ServerError) as e:
            code = getattr(e, "code", None)
            if code not in (429, 503):
                raise
            delay = _RETRY_DELAYS[attempt] if attempt < len(_RETRY_DELAYS) else _RETRY_DELAYS[-1]
            attempt += 1
            log.warning(
                "Gemini %d %s — retrying in %ds (attempt %d)...",
                code, type(e).__name__, delay, attempt,
            )
            time.sleep(delay)

# Per-token pricing in USD. Update when Gemini pricing changes.
# Source: https://ai.google.dev/pricing
_PRICING: dict[str, tuple[float, float]] = {
    "gemini-2.5-flash": (0.30 / 1_000_000, 0.30 / 1_000_000),
    "gemini-2.5-pro":   (1.25  / 1_000_000, 10.0  / 1_000_000),
    "gemini-2.0-flash": (0.10  / 1_000_000, 0.40  / 1_000_000),
}


def _estimate_cost(model_name: str, input_tokens: int, output_tokens: int) -> float | None:
    for prefix, (in_price, out_price) in _PRICING.items():
        if model_name.startswith(prefix):
            return input_tokens * in_price + output_tokens * out_price
    return None


def _build_context(chunks: list[RetrievedChunk]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, 1):
        citation = chunk.metadata.get("citation_en", chunk.doc_id)
        parts.append(f"[{i}] ({citation})\n{chunk.text}")
    return "\n\n".join(parts)


def _get_api_key(api_key: str | None, caller: str) -> str:
    import os
    key = api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not key:
        raise EnvironmentError(
            f"No Google API key found. Set GOOGLE_API_KEY or GEMINI_API_KEY, "
            f"or pass api_key= to {caller}."
        )
    return key


class GeminiGenerator(BaseGenerator):
    """
    Generates answers using Gemini 2.5 Flash.

    Parameters
    ----------
    model_name:
        Gemini model ID, default "gemini-2.5-flash".
    system_prompt:
        Instruction prepended to every request.
    api_key:
        Google API key. Falls back to GOOGLE_API_KEY / GEMINI_API_KEY env vars.
    temperature:
        Sampling temperature (0.0 = greedy).
    max_output_tokens:
        Upper limit on generated tokens.
    """

    def __init__(
        self,
        model_name: str = "gemini-2.5-flash",
        system_prompt: str = ANSWER_SYSTEM_PROMPT,
        api_key: str | None = None,
        temperature: float = 0.0,
        max_output_tokens: int = 1024,
    ):
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self._api_key = api_key
        self._client = None
        self._call_count: int = 0
        self._total_input_tokens: int = 0
        self._total_output_tokens: int = 0
        self._total_cost: float | None = None

    def _load(self) -> None:
        if self._client is not None:
            return
        from google import genai
        self._client = genai.Client(api_key=_get_api_key(self._api_key, "GeminiGenerator"))

    def _track_and_log(self, in_tok: int, out_tok: int) -> None:
        call_cost = _estimate_cost(self.model_name, in_tok, out_tok)
        self._call_count += 1
        self._total_input_tokens += in_tok
        self._total_output_tokens += out_tok
        if call_cost is not None:
            self._total_cost = (self._total_cost or 0.0) + call_cost
        call_cost_str = f"{call_cost:.6f}" if call_cost is not None else "N/A"
        total_cost_str = f"{self._total_cost:.6f}" if self._total_cost is not None else "N/A"
        log.info(
            "GeminiGenerator model=%s | call %d: input_tokens=%d output_tokens=%d cost_usd=%s"
            " | running total: input_tokens=%d output_tokens=%d cost_usd=%s",
            self.model_name, self._call_count, in_tok, out_tok, call_cost_str,
            self._total_input_tokens, self._total_output_tokens, total_cost_str,
        )

    def log_usage_summary(self) -> None:
        total_cost_str = f"{self._total_cost:.6f}" if self._total_cost is not None else "N/A"
        log.info(
            "GeminiGenerator usage summary | model=%s | calls=%d"
            " | total_input_tokens=%d | total_output_tokens=%d | total_cost_usd=%s",
            self.model_name, self._call_count,
            self._total_input_tokens, self._total_output_tokens, total_cost_str,
        )

    def generate(self, query: str, context_chunks: list[RetrievedChunk]) -> str:
        from google.genai import types
        self._load()
        context = _build_context(context_chunks)
        prompt = f"Context:\n{context}\n\nQuestion: {query}"
        response = _generate_with_retry(
            self._client,
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=self.system_prompt,
                temperature=self.temperature,
                max_output_tokens=self.max_output_tokens,
            ),
        )
        usage = response.usage_metadata
        self._track_and_log(usage.prompt_token_count, usage.candidates_token_count)
        return response.text


class GeminiJudge:
    """
    Evaluates a (query, generated_answer, reference_answer) triple using
    Gemini 2.5 Pro as an LLM-as-a-judge.

    Returns a dict with keys: faithfulness, correctness, completeness, rationale.
    """

    def __init__(
        self,
        model_name: str = "gemini-2.5-pro",
        api_key: str | None = None,
        temperature: float = 0.0,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self._api_key = api_key
        self._client = None
        self._call_count: int = 0
        self._total_input_tokens: int = 0
        self._total_output_tokens: int = 0
        self._total_cost: float | None = None

    def _load(self) -> None:
        if self._client is not None:
            return
        from google import genai
        self._client = genai.Client(api_key=_get_api_key(self._api_key, "GeminiJudge"))

    def _track_and_log(self, in_tok: int, out_tok: int) -> None:
        call_cost = _estimate_cost(self.model_name, in_tok, out_tok)
        self._call_count += 1
        self._total_input_tokens += in_tok
        self._total_output_tokens += out_tok
        if call_cost is not None:
            self._total_cost = (self._total_cost or 0.0) + call_cost
        call_cost_str = f"{call_cost:.6f}" if call_cost is not None else "N/A"
        total_cost_str = f"{self._total_cost:.6f}" if self._total_cost is not None else "N/A"
        log.info(
            "GeminiJudge model=%s | call %d: input_tokens=%d output_tokens=%d cost_usd=%s"
            " | running total: input_tokens=%d output_tokens=%d cost_usd=%s",
            self.model_name, self._call_count, in_tok, out_tok, call_cost_str,
            self._total_input_tokens, self._total_output_tokens, total_cost_str,
        )

    def log_usage_summary(self) -> None:
        total_cost_str = f"{self._total_cost:.6f}" if self._total_cost is not None else "N/A"
        log.info(
            "GeminiJudge usage summary | model=%s | calls=%d"
            " | total_input_tokens=%d | total_output_tokens=%d | total_cost_usd=%s",
            self.model_name, self._call_count,
            self._total_input_tokens, self._total_output_tokens, total_cost_str,
        )

    def judge(self, query: str, generated_answer: str, reference_answer: str) -> dict:
        """
        Returns
        -------
        dict with keys: faithfulness (1-5), correctness (1-5),
                        completeness (1-5), rationale (str).
        """
        import json
        from google.genai import types
        self._load()
        prompt = (
            f"Question: {query}\n\n"
            f"Generated answer: {generated_answer}\n\n"
            f"Reference answer: {reference_answer}"
        )
        response = _generate_with_retry(
            self._client,
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=JUDGE_SYSTEM_PROMPT,
                temperature=self.temperature,
                response_mime_type="application/json",
            ),
        )
        usage = response.usage_metadata
        self._track_and_log(usage.prompt_token_count, usage.candidates_token_count)
        return json.loads(response.text)


if __name__ == "__main__":
    import os
    from benchmark_rag.components.base import RetrievedChunk

    if not os.environ.get("GOOGLE_API_KEY") and not os.environ.get("GEMINI_API_KEY"):
        print("GOOGLE_API_KEY / GEMINI_API_KEY not set — skipping Gemini generator/judge test.")
    else:
        fake_chunks = [
            RetrievedChunk(
                text=(
                    "The trial judge found that the warrantless search of the appellant's "
                    "business premises violated s. 8 of the Charter. The documents seized "
                    "were excluded under s. 24(2) as their admission would bring the "
                    "administration of justice into disrepute."
                ),
                doc_id="2022 ONCA 100",
                chunk_idx=0,
                metadata={"citation_en": "2022 ONCA 100"},
                score=0.91,
            ),
            RetrievedChunk(
                text=(
                    "An office manager does not have actual or apparent authority to waive "
                    "an employer's Charter rights. Consent to search must come from someone "
                    "with actual authority over the premises and knowledge of the right being waived."
                ),
                doc_id="2022 ONCA 100",
                chunk_idx=1,
                metadata={"citation_en": "2022 ONCA 100"},
                score=0.87,
            ),
        ]
        query = "Can an office manager consent to a warrantless search on behalf of their employer?"

        # --- Generator ---
        print("Testing GeminiGenerator (gemini-2.5-flash) ...")
        generator = GeminiGenerator()
        answer = generator.generate(query, fake_chunks)
        print(f"\nQuery : {query}")
        print(f"Answer: {answer}")
        generator.log_usage_summary()

        # --- Judge ---
        reference = (
            "No. An office manager lacks authority to waive Charter rights on behalf of their employer. "
            "Such consent must come from someone with actual authority and knowledge of the right being waived."
        )
        print("\nTesting GeminiJudge (gemini-2.5-pro) ...")
        judge = GeminiJudge()
        scores = judge.judge(query, answer, reference)
        print(f"Judge scores: {scores}")
        judge.log_usage_summary()
