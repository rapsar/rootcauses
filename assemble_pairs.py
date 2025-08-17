"""
Assemble pairs between items from the physics vocabulary, using smart (OpenAI) LLMs
"""
import os
import json
import random
from pathlib import Path
from tqdm import tqdm
from openai import OpenAI
import itertools


def assemble_pairs_related_to(
    input_file: str,
    output_file: str,
    model: str = "gpt-4.1",
    max_pairs: int = 10
) -> None:
    """
    Assemble contrastive pairs between quantities (constants or variables) and concepts
    labeled with a related_to field.
    """

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Read input file
    quantities, concepts = [], []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            attrs = obj.get("attributes", {})
            if attrs.get("is_constant") or attrs.get("is_variable"):
                quantities.append(obj["item"])
            if attrs.get("is_concept") and not (attrs.get("is_constant") or attrs.get("is_variable")):
                concepts.append(obj["item"])

    print(f"Found {len(quantities)} quantities and {len(concepts)} concepts")

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build all candidate pairs and sample
    all_pairs = [(q, c) for q in quantities for c in concepts]
    sampled_pairs = random.sample(all_pairs, min(max_pairs, len(all_pairs)))

    with open(output_file, "a", encoding="utf-8") as out_f:
        for q, c in tqdm(sampled_pairs, desc="Processing pairs"):
            prompt = f"""
            Given the following physics terms:

            Quantity: "{q}"
            Concept: "{c}"

            Question: Is this quantity directly related to this concept in physics?
            Answer strictly in JSON with the schema:
            {{"related_to": true}} or {{"related_to": false}}
            """

            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a physics expert labeling relatedness between terms."},
                    {"role": "user", "content": prompt},
                ],
                max_completion_tokens=50,
                temperature=0,
            )

            content = response.choices[0].message.content.strip()
            try:
                related_obj = json.loads(content)
                related_to = bool(related_obj.get("related_to"))
            except Exception:
                print(f"Warning: Could not parse response for pair ({q}, {c}), skipping.")
                continue

            pair_obj = {"quantity": q, "concept": c, "related_to": related_to}
            out_f.write(json.dumps(pair_obj, ensure_ascii=False) + "\n")

    print(f"Pairs written to {output_file}: {len(sampled_pairs)}")
    
    
def assemble_pairs_hierarchy(
    input_file: str,
    output_file: str,
    model: str = "gpt-4.1",
    max_pairs: int = 200
) -> None:
    """
    Build pairs of principles, query LLM for their hierarchical relation, and write results as JSONL.
    """

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Read input file and extract principles
    principles = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            attrs = obj.get("attributes", {})
            if attrs.get("is_principle") is True:
                principles.append(obj["item"])

    print(f"Found {len(principles)} principles")
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build all unique pairs (p1, p2) where p1 != p2, no duplicates
    all_pairs = list(itertools.combinations(principles, 2))
    sampled_pairs = random.sample(all_pairs, min(max_pairs, len(all_pairs)))

    with open(output_file, "w", encoding="utf-8") as out_f:
        for p1, p2 in tqdm(sampled_pairs, desc="Processing principle pairs"):
            prompt = f"""
            Given the following two physics principles:

            Principle A: "{p1}"
            Principle B: "{p2}"

            Question: Are these two principles unrelated, or if related, which is more fundamental? Output strictly in JSON with the schema:
            {{"relation": "unrelated"}} or {{"relation": "A_more_fundamental"}} or {{"relation": "B_more_fundamental"}}
            Only use one of the three allowed values.
            """

            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a physics expert labeling hierarchical relations between principles."},
                    {"role": "user", "content": prompt},
                ],
                max_completion_tokens=50,
                temperature=0,
            )

            content = response.choices[0].message.content.strip()
            try:
                relation_obj = json.loads(content)
                relation = relation_obj.get("relation")
                if relation not in ("unrelated", "A_more_fundamental", "B_more_fundamental"):
                    print(f"Warning: Unexpected relation value '{relation}' for pair ({p1}, {p2}), skipping.")
                    continue
            except Exception:
                print(f"Warning: Could not parse response for pair ({p1}, {p2}), skipping.")
                continue

            pair_obj = {"principle_1": p1, "principle_2": p2, "relation": relation}
            out_f.write(json.dumps(pair_obj, ensure_ascii=False) + "\n")

    print(f"Principle pairs written to {output_file}: {len(sampled_pairs)}")


if __name__ == "__main__":
    assemble_pairs_related_to(
        input_file="data/physics_vocabulary_with_attributes_and_description.jsonl",
        output_file="data/physics_pairs.jsonl",
        model="gpt-4.1",
        max_pairs = 2048
    )
    
    # assemble_pairs_hierarchy(
    #     input_file="data/physics_vocabulary_with_attributes_and_description.jsonl",
    #     output_file="data/physics_pairs_hierarchy_41.jsonl",
    #     model="gpt-4.1",
    #     max_pairs = 1024
    # )