"""Persona style descriptors used by the query generator.

Each descriptor maps Big Five personality trait levels (low/medium/high) to
concrete query-style instructions that are injected into the generation prompt.
`LayPersonDescriptor` and `LegalExpertDescriptor` override a few mappings to
better match how each user type actually phrases legal search queries.
"""

from abc import ABC, abstractmethod


class UserDescriptor(ABC):
    @abstractmethod
    def get_text(self, *args, **kwargs) -> str:
        pass


class BaseFiveDescriptor(UserDescriptor):
    def __init__(
        self,
        openness="medium",
        conscientiousness="medium",
        extraversion="medium",
        agreeableness="medium",
        neuroticism="medium",
    ):
        super().__init__()
        self.traits = {
            "openness": openness,
            "conscientiousness": conscientiousness,
            "extraversion": extraversion,
            "agreeableness": agreeableness,
            "neuroticism": neuroticism,
        }
        self.levels = ["high", "medium", "low"]
        self.mappings = self.base_texts()

    def base_texts(self):
        return {
            ("conscientiousness", "high"):
                "Use perfect grammar, precise legal terminology, and complex sentence structures.",
            ("conscientiousness", "medium"):
                "Use standard, functional grammar.",
            ("conscientiousness", "low"):
                "Use loose grammar, potential typos, simple keywords, or fragment sentences.",

            ("extraversion", "high"):
                "Be conversational, wordy, or frame it as a natural language request.",
            ("extraversion", "low"):
                "Be terse, brief, and to the point.",

            ("neuroticism", "high"):
                "Tone should be urgent, worried, seeking reassurance, or focusing on risks/penalties.",
            ("neuroticism", "low"):
                "Tone should be calm, detached, and objective.",

            ("openness", "high"):
                "Focus on abstract concepts, reasoning, implications, or 'why' questions.",
            ("openness", "low"):
                "Focus strictly on concrete facts, dates, numbers, and specific outcomes.",

            ("agreeableness", "high"):
                "Phrasing should be polite, cooperative, and soft (e.g., 'Could you please help me find...').",
            ("agreeableness", "low"):
                "Phrasing should be demanding, skeptical, or aggressive (e.g., 'Prove that...').",
        }

    def opening_text(self):
        return "Do NOT cite specific case names, statute numbers, or legal tests."

    def get_text(self):
        desc = "User traits: " + ", ".join(f"{k}: {v}" for k, v in self.traits.items()) + "."
        style_hints = [self.opening_text()]
        for key in self.mappings:
            style_hints.append(self.mappings[key])
        return f"{desc} Query Style Instructions: {' '.join(style_hints)}"


class LayPersonDescriptor(BaseFiveDescriptor):
    def __init__(self, **trait_levels):
        super().__init__(**trait_levels)
        self.mappings.update({
            ("openness", "low"): (
                "Focus strictly on the tangible 'Adverse Event' or 'Injury' described in the text "
                "(e.g., 'forced to retake class', 'denied refund', 'basement flooded'). "
                "Do not use abstract summaries like 'school problem' or 'career help'."
            ),
            ("extraversion", "high"): (
                "Frame the query as a 'rambling' or 'venting' personal story. "
                "Include specific details of what went wrong to give context, "
                "rather than just asking the legal question directly."
            ),
            ("agreeableness", "high"): (
                "Use polite, soft openers and tone (e.g., 'I'm hoping you can help...'), "
                "but ensure the core complaint describes the specific unfair event mentioned in the text."
            ),
        })


class LegalExpertDescriptor(BaseFiveDescriptor):
    def __init__(self, **trait_levels):
        super().__init__(**trait_levels)
        self.mappings.update({
            ("extraversion", "low"):
                "Use formal legal query style with precise legal terminology "
                "(e.g., 'elements of negligence in tort law').",
            ("openness", "high"):
                "Focus on underlying legal principles, doctrines, or policy considerations "
                "(e.g., 'public policy rationale behind strict liability in product defects', "
                "'test for admissibility of wiretap evidence').",
            ("openness", "low"):
                "Focus on specific statutes, case law, or procedural rules "
                "(e.g., 'limitations period for debt recovery ontario').",
            ("conscientiousness", "low"):
                "Use legal shorthand or abbreviated terms (e.g., 'wrongful dismissal quantum calc').",
        })


if __name__ == "__main__":
    print(LegalExpertDescriptor(extraversion="low", openness="high").get_text())
    print()
    print(LayPersonDescriptor(extraversion="high", neuroticism="high").get_text())