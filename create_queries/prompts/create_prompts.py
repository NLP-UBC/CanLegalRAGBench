"""Persona-conditioned query generation from court decisions.

`DocumentQueryGenerator` generates one query per user persona (layperson and
legal associate) for a given decision. Each query is conditioned on a randomly
sampled Big Five trait profile (see `user_descriptions.py`) and a randomly
assigned target section of the document (Overview / Reasoning / Decision).
"""

import random

import dspy
import numpy as np

from prompts.user_descriptions import LayPersonDescriptor, LegalExpertDescriptor


class GenerateSyntheticData(dspy.Signature):
    """
    Given a court decision and a specific user persona/intent and target section of the document (Overview, Reasoning, Decision), generate a realistic search query that a user might type, and the corresponding factual answer found strictly within the document around the target section.

    The generated query must be a realistic search query or question relevant to the text but assuming no knowledge of the document. The query should be asked from the point of view of a user doing exploratory research on their issue. Do NOT reference specific names, dates, entities, or other specific case details. Create queries that a user might ask *without* knowing which case document contains the answer. The queries should reflect the legal issues, principles, fact patterns, and situations discussed in the provided court decision, but MUST NOT reveal or directly quote the document.

    The Query must have at least 2-3 unique facts or distinguishing details that are relevant to the document but do not directly reveal the document. For example, if the document is about a car accident case, the query could be "What happens if on a snowy day where the conditions were poor, I'm involved in a rear-end collision and the other driver was texting?" This query is realistic and relevant to the topic of car accidents, but it does not directly quote or reveal specific details from the document.

    Introduce variety in the types of queries generated, including but not limited to:
    - Hypothetical situations ("If I…", "What happens when…?")
    - Conceptual legal query ("How does the court determine…?")
    - Outcome-oriented query ("Would this count as…?")
    - Multi-step or scenario-based query requiring reasoning.
    - The query should be moderately complex, not trivial.
    """

    court_decision_text = dspy.InputField(
        desc="A full Canadian legal court decision."
    )
    user_persona = dspy.InputField(
        desc="The type of user using a Legal AI tool and their intent (e.g., 'Layperson looking for the outcome', 'Lawyer looking for citations')."
    )
    user_traits = dspy.InputField(
        desc="A description of the user's personality."
    )
    target_section = dspy.InputField(
        desc="The section of the document to focus on: Overview, Reasoning, Decision."
    )
    generated_query = dspy.OutputField(
        desc="A realistic search query that the user might type into a Legal AI tool, based on their persona and the target section."
    )


class DocumentQueryGenerator(dspy.Module):
    """Generates one query per persona for a single court decision."""

    def __init__(self):
        super().__init__()
        self.generate = dspy.Predict(GenerateSyntheticData)

        # Personas and their short identifiers. One query is generated per persona.
        self.user_types = {
            "Layperson: A non-legal expert asking a general query about a realistic legal scenario they may face in the real world. They do not use legal jargon.": "layperson",
            "Legal Associate: A lawyer searching through a Canadian case database looking for case decisions related to their current case.": "legal_associate",
        }
        self.trait_names = [
            "openness",
            "conscientiousness",
            "extraversion",
            "agreeableness",
            "neuroticism",
        ]
        self.levels = ["low", "medium", "high"]
        self.target_sections = ["Overview", "Reasoning", "Decision"]

    def _generate_random_persona_traits(self, role: str):
        """Sample a random Big Five profile and render it as style instructions."""
        trait_levels = {trait: random.choice(self.levels) for trait in self.trait_names}

        if role == "layperson":
            descriptor = LayPersonDescriptor(**trait_levels)
        elif role == "legal_associate":
            descriptor = LegalExpertDescriptor(**trait_levels)
        else:
            raise ValueError(f"unknown user type: {role}, expected ('layperson', 'legal_associate')")

        return descriptor.get_text(), trait_levels

    def forward(self, decision_chunk: str):
        results = []
        # Assign a distinct target section to each persona.
        target_sections = np.random.choice(self.target_sections, len(self.user_types), replace=False).tolist()

        for i, user_type in enumerate(self.user_types):
            short_user_type = self.user_types[user_type]
            target_section = target_sections[i]
            try:
                user_traits_as_str, traits = self._generate_random_persona_traits(role=short_user_type)
                prediction = self.generate(
                    court_decision_text=decision_chunk,
                    user_persona=user_type,
                    user_traits=user_traits_as_str,
                    target_section=target_section,
                )

                usage, cost = {}, 0.0
                current_lm = dspy.settings.lm
                if current_lm.history:
                    last_call = current_lm.history[-1]
                    usage = last_call.get("usage", {})
                    cost = last_call.get("cost", 0.0)

                results.append({
                    "user_type": short_user_type,
                    "user_traits": user_traits_as_str,
                    "target_section": target_section,
                    "query": prediction.generated_query,
                    "context_snippet": decision_chunk[:300] + "...",
                    "output_tokens": usage.get("completion_tokens", 0),
                    "input_tokens": usage.get("prompt_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                    "cost": cost,
                    **traits,
                })
            except Exception as e:
                print(f"Error generating for user_type '{short_user_type}' and section '{target_section}': {e}")

        return dspy.Prediction(results=results)