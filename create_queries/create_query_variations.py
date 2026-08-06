"""Step 3: Expand each seed query into a block of variations.

For each seed query (the filtered layperson queries from step 2):
1. The seed itself is judged (and minimally modified if rejected) and kept as
   fact variation #1.
2. One alternate-fact variation is generated: same scenario type and
   tone/style, but altered facts so different case law becomes relevant.
3. For each fact variation, 3 situation variations are generated (1 in a
   layperson tone, 2 in a legal-expert tone), each drawn from a pool of
   situation types (change point of view, next-steps focus, rule focus, ...)
   excluding the type the fact variation already matches.
Every generated variation passes through a judge-and-modify step that enforces
clarity/complexity criteria and labels an area of law.

This yields 8 rows per seed query: 2 fact variations x (1 base + 3 situation
variations). Each block of 8 rows is assigned one random Canadian province or
territory (used as retrieval context by the benchmark).

Usage:
    python create_query_variations.py --experiment-name query_variations \
        --input-csvs outputs/caseway_run/readable_judged_validation_set_caseway.csv \
                     outputs/york_run/readable_judged_validation_set_york.csv
"""

import argparse
import os
import random
from pathlib import Path

import dspy
import pandas as pd
from dotenv import load_dotenv

random.seed(42)

ALL_PROVINCES_AND_TERRITORIES = [
    "British Columbia",
    "Alberta",
    "Saskatchewan",
    "Manitoba",
    "Ontario",
    "Quebec",
    "New Brunswick",
    "Nova Scotia",
    "Prince Edward Island",
    "Newfoundland and Labrador",
    "Northwest Territories",
    "Yukon",
    "Nunavut",
]


class GenerateQueryVariation(dspy.Signature):
    """
    Creating queries to evaluate a RAG tool that helps everyday people navigate legal issues.
    The queries will be annotated with legal experts and are thus costly to annotate. To reduce cost, the variations should require minimal changes to the court case documents that are required to answer the query. The variations should not be so outlandish as to create unrealistic queries that people would never actually ask about legal issues. The variations should also not change the tone and style of the query unless specified to be in the tone of a legal expert.

    Receives a query and variation type and tone/style. Then creates a variation of the query.
    """

    query: str = dspy.InputField(description="The original query for which to generate variations.")
    query_variation_type: str = dspy.InputField(description="The type of variation to create")
    variation_description: str = dspy.InputField(description="A description of the type of variation to create")
    query_variation_example: str = dspy.InputField(description="An example of the type of variation to create")
    original_query_example: str = dspy.InputField(description="An example of an original query that the variation example is based on")
    query_variation_tone_style: str = dspy.InputField(
        description="The tone and style to use when creating the query variation. For example, 'layperson' or 'legal expert'."
    )
    query_variation: str = dspy.OutputField(
        description="The generated query variation based on the input query, variation type, and tone/style."
    )


class GenerateAlternateFactVariation(dspy.Signature):
    """
    Creating queries to evaluate a RAG tool that helps everyday people navigate legal issues.
    Receives a query and generates a variation on the facts of the query while keeping the same scenario type and tone/style. The fact variation should be a realistic alteration of the original facts that still fits within the same general scenario type.
    The query variation should change what court cases would be relevant to the query, but should not be so outlandish as to change the scenario type. The variations should also not change the tone and style of the query.
    The query variation should not increase the complexity of the query or make it more difficult to understand. The query variation should be something that a person might realistically ask about in relation to the same type of legal issue.

    Examples:
    1.  Original query: "My kid, he's 8, got into a fight and punched someone just one time. But the person got seriously hurt. Now we're dealing with court. Will he go to jail? Like, what kind of punishment could he get for causing bad injuries from one punch? I'm so worried."
        Fact variation example: "My kid, he's 17, got into a fight and punched someone just one time. But the person got seriously hurt. Now we're dealing with court. Will he go to jail? Like, what kind of punishment could he get for causing bad injuries from one punch? I'm so worried."

    2.  Original query: "My ex keeps making up stories that I hurt our child, even when all the investigations show nothing happened. It's so frustrating because they keep involving child services and now everyone thinks *her* constant talk about abuse is actually making our child suffer emotionally. What happens if one parent won't stop pushing false allegations and it causes harm to the child, and what can be done to protect the child from their other parent's actions like that?"
        Fact variation example: "My ex keeps making up stories that I hurt our child. It's so frustrating because they keep involving child services and now everyone thinks *my* constant talk about abuse is actually making our child suffer emotionally. What happens if one parent won't stop pushing false allegations and it causes harm to the child, and what can be done to protect the child from their other parent's actions like that?"
    """

    original_query: str = dspy.InputField(description="The original query for which to generate a fact variation.")
    query_variation: str = dspy.OutputField(
        description="The generated query variation based on the input original query with altered facts but the same scenario type and tone/style."
    )


class JudgeAndModifyQueryVariation(dspy.Signature):
    """
    Receives a query variation and judges whether it is a good variation to include in the RAG validation set based on the criteria of being unambiguous (A legal student should be able to identify the legal issue).

    The queries are written to be used by a general public, some lacking in legal knowledge, while others are written to be used by someone with more legal knowledge. Using some vague terms is intended, but they should not be so vague or ambiguous that it is not clear what the legal issue is.
    For example, "What are my rights if my landlord enters my apartment without notice?" is a good query because it is clear that the legal issue is about tenant rights and landlord entry without notice, while "I have a problem with my landlord, what can I do?" is too vague.

    It should also not be so simple or unserious that someone would be comfortable just asking Google and using the first thing that pops up. For example "How do I file for divorce?" is too simple and generic, while "What are the grounds for divorce if my spouse has been emotionally abusive but not physically abusive?" is more specific and complex.
    It should also not be too long or complex such that it would either require expert legal knowledge to understand, or searching through excessive amounts of legal documents to answer.
    If the query variation is rejected, modify the query variation to try to fix the issues that caused it to be rejected and return the modified query variation.
    The modifications should be as minimal as possible and should try to keep the same scenario type and tone/style.

    The judge should also identify an area of law that the query variation belongs to.
    """

    query_variation: str = dspy.InputField(description="The query variation to judge and modify if necessary.")
    final_verdict: str = dspy.OutputField(
        description="The final verdict of whether the query variation is a good variation to include in the RAG validation set based on the criteria. Must be exactly 'Keep' or 'Reject'."
    )
    reasoning: str = dspy.OutputField(
        description="The reasoning behind the final verdict and modifications made to the query variation if the final verdict is 'Reject'."
    )
    modified_query_variation: str = dspy.OutputField(
        description="If the final verdict is 'Reject', modify the query variation to try to fix the issues that caused it to be rejected and return the modified query variation. The modifications should be as minimal as possible and should try to keep the same scenario type and tone/style."
    )
    law_area: str = dspy.OutputField(
        description="An area of law that the query variation belongs to. For example, ['family law', 'tenant law', 'employment law', 'criminal law', 'tax law', 'immigration law', 'intellectual property law', 'contract law', 'tort law', 'constitutional law', 'administrative law', 'environmental law', 'human rights law', 'corporate law', 'bankruptcy law', 'international law']. If none apply, the judge can create another category."
    )


class QueryVariationIdentifier(dspy.Signature):
    """
    Receives a query and identifies the type of query variation that it is. The query variation types are:
    1. Change Point of View: Change the perspective to another party involved (e.g. landlord vs tenant, employee vs employer, etc.) while keeping the facts the same. Do not change to the perspective of an attorney, just the perspective of another party involved in the same legal issue.
    2. Next Steps Focus: Focus the query on the next steps or process rather than the legal principles. It can also be a question about how to find more information or get help
    3. Rule Focus: Focus the query on the legal rule or principle rather than the situation or process
    4. Information Seeking: Make the query more explicitly about seeking information or understanding, rather than asking for advice or next steps
    5. Burden of Proof: Focus the query on the burden of proof or evidence needed to support a claim
    6. General Interpretations of a Legal Principle: Make the query more general and focused on interpretations of a legal principle rather than a specific situation
    7. None: The query variation does not fit into any of the above categories.
    """

    query_variation: str = dspy.InputField(description="The query variation for which to identify the type.")
    query_variation_type: str = dspy.OutputField(
        description="The identified type of query variation. Must be one of the following: ['Change Point of View', 'Next Steps Focus', 'Rule Focus', 'Information Seeking', 'Burden of Proof', 'General Interpretations of a Legal Principle', 'None']"
    )


class QueryVariationCreator(dspy.Module):
    """
    Receives a query and generates 1 variation on the facts with the same type of scenario. For each fact
    variation (the judged original + the alternate-fact variation), generates 3 situation variations,
    1 in the tone and style of a layperson and 2 in the tone and style of someone more well versed in
    legal terminology.
    """

    def __init__(self, max_cost: float = 50.0):
        super().__init__()
        self.total_cost = 0.0
        self.max_cost = max_cost  # dollars

        self.generateAlternativeFactVariation = dspy.ChainOfThought(GenerateAlternateFactVariation)
        self.generateAlternativeSituationVariation = dspy.ChainOfThought(GenerateQueryVariation)
        self.judgeAndModifyQueryVariation = dspy.ChainOfThought(JudgeAndModifyQueryVariation)
        self.queryVariationIdentifier = dspy.ChainOfThought(QueryVariationIdentifier)
        self.num_tries_per_query = 2

        self.additional_num_fact_variations = 1
        self.num_laypeople_variations_per_fact_variation = 1
        self.num_legal_expert_variations_per_fact_variation = 2
        self.num_situation_variations_per_fact_variation = (
            self.num_laypeople_variations_per_fact_variation + self.num_legal_expert_variations_per_fact_variation
        )
        self.legal_knowledge_list = (
            ["layperson"] * self.num_laypeople_variations_per_fact_variation
            + ["legal expert"] * self.num_legal_expert_variations_per_fact_variation
        )

        self.original_query_example = "What are my rights if my landlord enters my apartment without notice?"
        # (name, description, example) triples for the situation-variation pool.
        self.situation_variations = [
            (
                "Change Point of View",
                "Change the perspective to another party involved (e.g. landlord vs tenant, employee vs employer, etc.) while keeping the facts the same. Do not change to the perspective of an attorney, just the perspective of another party involved in the same legal issue.",
                "For example, if the original query is from the perspective of a tenant asking about eviction, the variation could be from the perspective of a landlord asking about how to evict a tenant.",
            ),
            (
                "Next Steps Focus",
                "Focus the query on the next steps or process rather than the legal principles. It can also be a question about how to find more information or get help",
                "For example, if the original query is 'What are my rights if my landlord enters my apartment without notice?', the variation could be 'What can I do if my landlord enters my apartment without notice?'",
            ),
            (
                "Rule Focus",
                "Focus the query on the legal rule or principle rather than the situation or process",
                "For example, if the original query is 'What are my rights if my landlord enters my apartment without notice?', the variation could be 'What determines if a landlord needs to give notice before entering a tenant's apartment?'",
            ),
            (
                "Information Seeking",
                "Make the query more explicitly about seeking information or understanding, rather than asking for advice or next steps",
                "For example, if the original query is 'What are my rights if my landlord enters my apartment without notice?', the variation could be 'Do the courts generally require landlords to give notice before entering a tenant's apartment?'",
            ),
            (
                "Burden of Proof",
                "Focus the query on the burden of proof or evidence needed to support a claim",
                "For example, if the original query is 'What are my rights if my landlord enters my apartment without notice?', the variation could be 'What kind of evidence would I need to show that my landlord entered my apartment without notice?'",
            ),
            (
                "General Interpretations of a Legal Principle",
                "Make the query more general and focused on interpretations of a legal principle rather than a specific situation",
                "For example, if the original query is 'What are my rights if my landlord enters my apartment without notice?', the variation could be 'How do courts interpret the requirement for landlords to give notice before entering a tenant's apartment?'",
            ),
        ]

    def _track_last_call(self):
        """Read usage/cost of the most recent LM call and add it to the running total."""
        usage, cost = {}, 0.0
        current_lm = dspy.settings.lm
        if current_lm.history:
            last_call = current_lm.history[-1]
            usage = last_call.get("usage", {})
            cost = last_call.get("cost", 0.0)
        self.total_cost += cost
        return usage, cost

    def judge_and_modify_query_variation(self, query_variation: str):
        judge_result = self.judgeAndModifyQueryVariation(query_variation=query_variation)
        final_verdict = judge_result["final_verdict"]
        law_area = judge_result["law_area"]
        if "reject" in final_verdict.lower():
            print(f"Query variation rejected: {query_variation}\nReasoning: {judge_result['reasoning']}")
            modified_query_variation = judge_result["modified_query_variation"]
        else:
            modified_query_variation = query_variation

        num_tries = 1
        while "reject" in final_verdict.lower() and num_tries < self.num_tries_per_query:
            judge_result = self.judgeAndModifyQueryVariation(query_variation=modified_query_variation)
            final_verdict = judge_result["final_verdict"]
            law_area = judge_result["law_area"]
            if "reject" in final_verdict.lower():
                modified_query_variation = judge_result["modified_query_variation"]
            num_tries += 1

        return modified_query_variation, law_area

    def create_query_variations(self, query: str) -> list[dict]:
        results = []
        fact_variations = []
        fact_variation_law_areas = []

        # The judged (possibly minimally modified) original query is fact variation #1.
        og_query, law_area = self.judge_and_modify_query_variation(query)
        if og_query != query:
            print(f"Original query was modified by the judge. Original: {query}\n\nModified: {og_query}\n\n")
        fact_variations.append(og_query)
        fact_variation_law_areas.append(law_area)

        for _ in range(self.additional_num_fact_variations):
            fact_variation = self.generateAlternativeFactVariation(original_query=og_query).query_variation
            self._track_last_call()
            judged_fact_variation, law_area = self.judge_and_modify_query_variation(fact_variation)
            fact_variations.append(judged_fact_variation)
            fact_variation_law_areas.append(law_area)

        for fact_variation, law_area in zip(fact_variations, fact_variation_law_areas):
            if self.total_cost >= self.max_cost:
                print(f"Total cost limit of ${self.max_cost} reached. Stopping generation of query variations.")
                break

            # Base row: the fact variation itself, with no situation variation applied.
            results.append({
                "original_query": query,
                "query": fact_variation,
                "fact_variation": fact_variation,
                "situation_variation": fact_variation,
                "law_area": law_area,
                "situation_variation_type": "None",
                "situation_variation_description": "None",
                "situation_variation_example": "None",
                "query_variation_tone_style": "None",
                "output_tokens": 0,
                "input_tokens": 0,
                "total_tokens": 0,
                "cost": 0.0,
            })

            # Exclude the situation type the fact variation already matches, for diversity.
            og_query_variation_type = self.queryVariationIdentifier(query_variation=fact_variation)["query_variation_type"]
            self._track_last_call()
            candidate_variations = [v for v in self.situation_variations if v[0] != og_query_variation_type]
            situation_sample = random.sample(
                candidate_variations,
                min(self.num_situation_variations_per_fact_variation, len(candidate_variations)),
            )

            for i, (situation_name, situation_description, situation_example) in enumerate(situation_sample):
                query_variation = self.generateAlternativeSituationVariation(
                    query=fact_variation,
                    query_variation_type=situation_name,
                    variation_description=situation_description,
                    query_variation_example=situation_example,
                    original_query_example=self.original_query_example,
                    query_variation_tone_style=self.legal_knowledge_list[i],
                ).query_variation
                usage, cost = self._track_last_call()

                judged_query_variation, judged_law_area = self.judge_and_modify_query_variation(query_variation)

                results.append({
                    "original_query": query,
                    "query": judged_query_variation,
                    "fact_variation": fact_variation,
                    "situation_variation": judged_query_variation,
                    "law_area": judged_law_area,
                    "situation_variation_type": situation_name,
                    "situation_variation_description": situation_description,
                    "situation_variation_example": situation_example,
                    "query_variation_tone_style": self.legal_knowledge_list[i],
                    "output_tokens": usage.get("completion_tokens", 0),
                    "input_tokens": usage.get("prompt_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                    "cost": cost,
                })

        return results


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-exp", "--experiment-name", required=True, help="name for this run; outputs go to outputs/<experiment-name>/")
    parser.add_argument("-in", "--input-csvs", type=Path, nargs="+", required=True, help="CSVs of filtered seed queries (must have a 'query' column)")
    parser.add_argument("--model", default="gemini/gemini-2.5-flash", help="LiteLLM model string")
    parser.add_argument("--max-cost", type=float, default=50.0, help="stop generating once total cost exceeds this many dollars")
    parser.add_argument("--save-every", type=int, default=25, help="save intermediate results every N seed queries")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="base output directory")
    args = parser.parse_args()

    load_dotenv()
    lm = dspy.LM(args.model, api_key=os.getenv("GEMINI_API_KEY"), temperature=0.0)
    dspy.configure(lm=lm, track_usage=True)

    queries = []
    for source_path in args.input_csvs:
        df = pd.read_csv(source_path)
        queries.extend(df["query"].tolist())
    print(f"Total seed queries collected: {len(queries)}")

    output_dir = args.output_dir / args.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    creator = QueryVariationCreator(max_cost=args.max_cost)
    final_queries = []
    query_provinces = []

    for i, query in enumerate(queries):
        query_variations = creator.create_query_variations(query)
        final_queries.extend(query_variations)
        # One random province per seed query, shared by its whole block of variations.
        query_provinces.extend([random.choice(ALL_PROVINCES_AND_TERRITORIES)] * len(query_variations))
        print(f"Cost after generating variations for query {i + 1}/{len(queries)}: ${creator.total_cost:.2f}\n")

        if i % args.save_every == 0:
            intermediate_df = pd.DataFrame(final_queries)
            intermediate_df["province"] = query_provinces
            intermediate_df["index"] = intermediate_df.index
            intermediate_path = output_dir / f"intermediate_query_variations_{i + 1}_queries.json"
            intermediate_df.to_json(intermediate_path, orient="records")
            print(f"Saved intermediate query variations to {intermediate_path}")

    print(f"Total query variations generated: {len(final_queries)}")

    df = pd.DataFrame(final_queries)
    df["province"] = query_provinces
    df["index"] = df.index
    output_path = output_dir / "query_variations.json"
    df.to_json(output_path, orient="records")
    print(f"Saved query variations to {output_path}")
    print(f"Total cost of generating query variations: ${creator.total_cost:.2f}")


if __name__ == "__main__":
    main()