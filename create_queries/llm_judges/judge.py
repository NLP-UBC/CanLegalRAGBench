"""LLM judge for (query, document) pairs.

`LegalJudge` audits a generated query against its source document and returns a
Keep/Reject verdict based on a modular list of criteria. Used by
`2_filter_by_judge.py` to filter generated queries before human annotation.
"""

from typing import List

import dspy


class Criterion:
    def __init__(self, name: str, description: str, bad_examples: List[str], good_examples: List[str]):
        self.name = name
        self.description = description
        self.bad_examples = bad_examples
        self.good_examples = good_examples

    def to_prompt_string(self) -> str:
        """Formats the criterion into a strict instruction block for the LLM."""
        bad_ex = "\n".join(f"      - [FAIL]: {ex}" for ex in self.bad_examples)
        good_ex = "\n".join(f"      - [PASS]: {ex}" for ex in self.good_examples)
        return (
            f"### {self.name}\n"
            f"   **Rule**: {self.description}\n"
            f"   **Examples**:\n{bad_ex}\n{good_ex}\n"
        )


BASIC_CRITERIA = [
    Criterion(
        name="Structural Integrity (No AI Artifacts)",
        description="The query must not contain 'AI-isms', references to 'the provided text', or explicit request for summaries. It should sound like a human asking a question, not a prompt engineer.",
        bad_examples=[
            "Summarize the provided legal document regarding the plaintiff.",
            "Based on the text below, what was the ruling?",
            "Extract the key dates from this case file.",
        ],
        good_examples=[
            "What is the precedent for constructive dismissal in Ontario?",
            "Can a landlord evict a tenant for personal use if they own another property?",
        ],
    ),
    Criterion(
        name="Information Gap (The 'Google Test')",
        description="The query must be specific enough to require a legal retrieval tool. If the query is trivial generic knowledge (answerable by a quick Google search without this doc), it fails.",
        bad_examples=[
            "What does the Supreme Court do?",
            "Define 'contract'.",
            "Who is the current Prime Minister?",
        ],
        good_examples=[
            "Does the Oakes test apply to administrative tribunals?",
            "limitations period for medical malpractice in British Columbia 2023",
        ],
    ),
    Criterion(
        name="Relevance & Grounding",
        description="The document must actually contain the answer or be highly relevant precedent for the query. The query must not hallucinate facts not present in the document.",
        bad_examples=[
            "Query asks about a 'drunk truck driver' when the document is about a 'slip and fall' in a grocery store.",
            "Query asks for the 2024 statute amendment, but the document is a 1990 case decision.",
        ],
        good_examples=[
            "Query asks about 'slip and fall liability' and the document is a ruling on 'Doe v. Walmart' regarding a wet floor.",
            "Query asks for 'grounds for patent invalidation' and the document outlines a decision on 'obviousness' in a patent suit.",
        ],
    ),
    Criterion(
        name="Tax court specific test for relevant cases",
        description="The document must be about self represented litigants, informal procedures, tax dispute resolution mechanisms or other disputes that are relevant to everyday people. This is specifically about the type of query and not about matching the query to the document. If the document is about a tax court case that is very technical and not relevant to everyday people, it fails.",
        bad_examples=[
            "The document is about a complex corporate tax dispute between two multinational corporations that has no relevance to everyday people.",
            "The document is about a tax court case that is very technical and not relevant to everyday people.",
        ],
        good_examples=[
            "The document is about a tax court case involving a self represented litigant disputing a CRA assessment for a small business.",
            "The document is about a tax court case involving an individual disputing the disallowance of a home office deduction.",
        ],
    ),
]


class LegalRetrievalJudgeSignature(dspy.Signature):
    """
    You are an expert Legal Retrieval Judge.
    You will audit a (Query, Document) pair to decide if it belongs in a Gold Standard Test Set for a realistic Legal Retrieval system that retrieves documents based on questions people have about the law or legal precedents.
    The QUERY is a user query that should be answerable by retrieving the DOCUMENT.
    The DOCUMENT is a legal court decision that serves as the ground truth document to retrieve in a Legal Retrieval system.

    You must evaluate the pair against the provided CRITERIA GUIDELINES.
    If the pair fails ANY of the criteria, the final verdict must be 'Reject'.
    """

    document_snippet = dspy.InputField(desc="The legal text serving as the ground truth answer.")
    query = dspy.InputField(desc="The generated user question to be evaluated.")
    criteria_guidelines = dspy.InputField(desc="The specific rules and examples this pair must satisfy.")

    analysis = dspy.OutputField(desc="A step-by-step analysis of how the pair meets or fails EACH criterion.")
    final_verdict = dspy.OutputField(desc="Must be exactly 'Keep' or 'Reject'.")


class LegalJudge(dspy.Module):
    def __init__(self, criteria_list: List[Criterion] = BASIC_CRITERIA, total_max_cost: float = 10.0):
        super().__init__()
        self.total_max_cost = total_max_cost  # dollars
        self.current_cost = 0.0
        self.criteria_list = criteria_list
        self.prog = dspy.ChainOfThought(LegalRetrievalJudgeSignature)

    def forward(self, query: str, document_snippet: str):
        if self.current_cost >= self.total_max_cost:
            raise RuntimeError(
                f"Total cost limit reached for LegalJudge module. "
                f"Current cost: {self.current_cost}, Max cost: {self.total_max_cost}"
            )

        criteria_str = "\n".join(c.to_prompt_string() for c in self.criteria_list)
        pred = self.prog(
            query=query,
            document_snippet=document_snippet,
            criteria_guidelines=criteria_str,
        )

        usage, cost = {}, 0.0
        current_lm = dspy.settings.lm
        if current_lm.history:
            last_call = current_lm.history[-1]
            usage = last_call.get("usage", {})
            cost = last_call.get("cost", 0.0)
            self.current_cost += cost

        result = {
            "query": query,
            "document_snippet": document_snippet,
            "analysis": pred.analysis,
            "final_verdict": pred.final_verdict,
            "judge_output_tokens": usage.get("completion_tokens", 0),
            "judge_input_tokens": usage.get("prompt_tokens", 0),
            "judge_total_tokens": usage.get("total_tokens", 0),
            "judge_cost": cost,
            "criteria": "Default Criteria",
        }
        return dspy.Prediction(results=[result])