"""
Centralised prompt strings for every LLM-driven component in the pipeline.

Each sibling module holds the prompt(s) for one logical role and, in a header
comment, names the consumer files that import it. This keeps prompt text out of
pipeline logic and makes it trivial to audit / diff prompt changes.

Import prompt constants directly from their module, e.g.:

    from benchmark_rag.prompts.answer_generator import ANSWER_SYSTEM_PROMPT
"""
