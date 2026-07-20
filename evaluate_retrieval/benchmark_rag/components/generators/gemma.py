"""Gemma 4 generator via HuggingFace Transformers."""
from __future__ import annotations

import logging

import torch
from benchmark_rag.components.base import BaseGenerator, RetrievedChunk
from benchmark_rag.prompts.answer_generator import ANSWER_SYSTEM_PROMPT

log = logging.getLogger(__name__)


def _build_context(chunks: list[RetrievedChunk]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, 1):
        citation = chunk.metadata.get("citation_en", chunk.doc_id)
        parts.append(f"[{i}] ({citation})\n{chunk.text}")
    return "\n\n".join(parts)


class GemmaGenerator(BaseGenerator):
    """
    Generates answers using Gemma 4 MoE (local HuggingFace model).

    Uses AutoTokenizer for text-only generation (AutoProcessor requires torchvision).

    Parameters
    ----------
    model_name:
        HuggingFace model ID, default "google/gemma-4-26B-A4B-it".
    system_prompt:
        Instruction prepended to every request.
    device:
        Torch device string, e.g. "cuda:0" or "auto".
    temperature:
        Sampling temperature (0.0 = greedy).
    max_new_tokens:
        Upper limit on generated tokens.
    """

    def __init__(
        self,
        model_name: str = "google/gemma-4-26B-A4B-it",
        system_prompt: str = ANSWER_SYSTEM_PROMPT,
        device: str = "auto",
        temperature: float = 0.0,
        max_new_tokens: int = 8000,
    ):
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.device = device
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self._model = None
        self._tokenizer = None
        self._call_count: int = 0
        self._total_input_tokens: int = 0
        self._total_output_tokens: int = 0

    def _load(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoModelForCausalLM, AutoTokenizer

        log.info("Loading Gemma model %s ...", self.model_name)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
        )
        log.info("Gemma model loaded.")

    def _track_and_log(self, in_tok: int, out_tok: int) -> None:
        self._call_count += 1
        self._total_input_tokens += in_tok
        self._total_output_tokens += out_tok
        log.info(
            "GemmaGenerator model=%s | call %d: input_tokens=%d output_tokens=%d"
            " | running total: input_tokens=%d output_tokens=%d",
            self.model_name, self._call_count, in_tok, out_tok,
            self._total_input_tokens, self._total_output_tokens,
        )

    def log_usage_summary(self) -> None:
        log.info(
            "GemmaGenerator usage summary | model=%s | calls=%d"
            " | total_input_tokens=%d | total_output_tokens=%d",
            self.model_name, self._call_count,
            self._total_input_tokens, self._total_output_tokens,
        )

    def generate(self, query: str, context_chunks: list[RetrievedChunk]) -> str:
        self._load()
        context = _build_context(context_chunks)
        user_message = f"Context:\n{context}\n\nQuestion: {query}"

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_message},
        ]
        input_ids = self._tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
        ).to(self._model.device)

        gen_kwargs = dict(max_new_tokens=self.max_new_tokens)
        if self.temperature == 0.0:
            gen_kwargs["do_sample"] = False
        else:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = self.temperature

        with torch.no_grad():
            output_ids = self._model.generate(input_ids, **gen_kwargs)

        input_len = input_ids.shape[-1]
        output_len = output_ids.shape[-1] - input_len
        self._track_and_log(input_len, output_len)

        return self._tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=True)


if __name__ == "__main__":
    from benchmark_rag.components.base import RetrievedChunk

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
    ]
    query = "Can an office manager consent to a warrantless search on behalf of their employer?"

    print(f"Testing GemmaGenerator ({GemmaGenerator.__name__}) ...")
    generator = GemmaGenerator(device="auto")
    answer = generator.generate(query, fake_chunks)
    print(f"\nQuery : {query}")
    print(f"Answer: {answer}")
    generator.log_usage_summary()
