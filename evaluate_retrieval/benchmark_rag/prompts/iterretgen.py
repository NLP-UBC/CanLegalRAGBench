"""
Intermediate-answer prompts for IterRetGen.

These drive the short summaries that get appended to the query between
retrieval iterations. They are NOT the final user-facing answer — the final
answer is produced by the regular answer generator using
`prompts/answer_generator.py`. So these prompts stay compact and focused on
producing useful query-augmentation text; they do not enforce the three-section
structure. They do, however, ask the model to flag aspects the passages do
not cover, which helps the next iteration search for what is missing.

Used by:
  - benchmark_rag/pipeline/iterretgen_pipeline.py
"""

SHORT_INTERMEDIATE_PROMPT = (
    "You are a legal research assistant drafting a brief intermediate note that "
    "will be used to refine the next retrieval round. Based ONLY on the provided "
    "context passages, write a short paragraph summarising what the passages say "
    "about the question. Focus on what courts have decided previously rather than "
    "giving a definitive yes / no. If the passages do not cover an aspect of the "
    "question, say so briefly — naming the missing aspect helps the next search."
)

FULL_INTERMEDIATE_PROMPT = (
    "You are a legal research assistant drafting an intermediate answer that will "
    "be used to refine the next retrieval round. Answer the question accurately "
    "and concisely using ONLY the provided context passages. Focus on what courts "
    "have decided previously rather than issuing a definitive yes / no. If the "
    "passages do not contain enough information for some aspect of the question, "
    "say so clearly and name the missing aspect so the next search can target it. "
    "Cite the relevant passage(s) when possible."
)
