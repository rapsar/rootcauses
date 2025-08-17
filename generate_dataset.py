"""
Generates physics-related vocabulary and attributes/description using OpenAI LLMs

TODO:
maybe use outlines from structured input
make API calls GPT-5 compatible
"""
import json
from openai import OpenAI
import os
from pathlib import Path
import outlines
from tqdm import tqdm


def generate_vocabulary(
    model: str = "gpt-5-mini", 
    prompt_file: str = "prompt_vocabulary.txt",
    num_items: int = 100,
    output_file: str = "physics_vocabulary.jsonl",
) -> None:
    """
    Generate a vocabulary of physics concepts using OpenAI's API.
    """
    
    # set up OpenAI client
    client = OpenAI(os.getenv("OPENAI_API_KEY"))
    
    # load prompt file
    with open(prompt_file, 'r', encoding='utf-8') as pf:
        prompt = pf.read()
    
    # tells how many
    prompt = f"Generate exactly {num_items} physics concepts/variables/constants/principles. \n" + prompt
    # print(prompt)

    try:
        print(f"Querying {model} for {num_items} physics concepts...")
        
        # call API
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a physics expert tasked with generating comprehensive vocabulary lists."},
                {"role": "user", "content": prompt}
            ],
            # max_completion_tokens=1000,
        )
        
        # exatract response content
        content = response.choices[0].message.content.strip()
        
        # parse JSON
        try:
            items = json.loads(content)
        except json.JSONDecodeError:
            # If direct JSON parsing fails, try to extract JSON from the response
            start_idx = content.find('[')
            end_idx = content.rfind(']') + 1
            if start_idx != -1 and end_idx != 0:
                items = json.loads(content[start_idx:end_idx])
            else:
                raise ValueError("Could not parse JSON from API response")
        
        # Validate the response structure
        if not isinstance(items, list):
            raise ValueError("API response is not a list of items")
        
        # Create output directory if it doesn't exist
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # write to jsonl
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in items:
                if not isinstance(item, str):
                    print(f"Warning: Skipping malformed item: {item}")
                    continue
                json.dump({"item": item}, f, ensure_ascii=False)
                f.write('\n')
        
        print(f"Successfully generated {len(items)} physics concepts")
        print(f"Results saved to: {output_file}")
            
    except Exception as e:
        print(f"Error generating vocabulary: {str(e)}")
        raise


def add_attributes_to_vocab(
    input_file: str,
    output_file: str,
    model: str = "gpt-4.1-mini"
) -> None:
    """
    Take every item in an existing .jsonl vocabulary file and ask an OpenAI model
    to classify it into four boolean attributes:
        - is_constant
        - is_variable
        - is_concept
        - is_principle
    Save a new .jsonl file with the extra attributes.
    """

    # set up client
    client = OpenAI(os.getenv("OPENAI_API_KEY"))

    # read from input file
    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as out_f:
        for line in tqdm(lines, desc="Processing items", unit="item"):
            obj = json.loads(line)
            item = obj.get("item")

            prompt = f"""
            Classify the following physics term into four boolean attributes.

            Term: "{item}"

            Return a JSON object with exactly these keys:
            - is_constant
            - is_variable
            - is_concept
            - is_principle

            Each value should be true or false.

            For example, if item is "speed of light" you should return:
            is_constant: True
            is_variable: False
            is_concept: False
            is_principle: False

            Only return the JSON object, nothing else.
            """

            # call API
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a physics expert tasked with classifying vocabulary terms."},
                    {"role": "user", "content": prompt}
                ],
                max_completion_tokens=200,
                temperature=0
            )

            content = response.choices[0].message.content.strip()

            try:
                attributes = json.loads(content)
            except json.JSONDecodeError:
                print(f"Warning: Could not parse attributes for {item}, skipping.")
                continue

            obj["attributes"] = attributes
            out_f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"Attribute-enriched vocabulary written to {output_file}")
 
    
def add_description_to_vocab(
    input_file: str,
    output_file: str,
    model: str = "gpt-4.1-mini"
) -> None:
    """
    Add a short description to each vocabulary item in a .jsonl file.
    """

    client = OpenAI(os.getenv("OPENAI_API_KEY"))

    # Read items from input file
    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as out_f:
        for line in tqdm(lines, desc="Adding descriptions", unit="item"):
            obj = json.loads(line)
            item = obj.get("item")

            prompt = f"""
            Provide a concise description (in a few words) of the following physics term:

            Term: "{item}"

            Return only a JSON object with this single key:
            - description
            """

            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a physics expert providing short descriptions of terms."},
                    {"role": "user", "content": prompt}
                ],
                max_completion_tokens=200,
                temperature=0
            )

            content = response.choices[0].message.content.strip()

            try:
                desc_obj = json.loads(content)
            except json.JSONDecodeError:
                print(f"Warning: Could not parse description for {item}, skipping.")
                continue

            obj["description"] = desc_obj.get("description", "")
            out_f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"Vocabulary with descriptions written to {output_file}")


if __name__ == "__main__":

    # generate_vocabulary(
    #     model="gpt-4.1",  
    #     prompt_file="prompt_vocabulary.txt",
    #     num_items=512,
    #     output_file="data/physics_vocabulary.jsonl"
    # )
    
    # add_attributes_to_vocab(
    #     input_file="data/physics_vocabulary.jsonl",
    #     output_file="data/physics_vocabulary_with_attributes.jsonl",
    #     model="gpt-4.1-mini" # don't use gpt-5 !!!
    # )
    
    add_description_to_vocab(
        input_file="data/physics_vocabulary_with_attributes.jsonl",
        output_file="data/physics_vocabulary_with_attributes_and_description.jsonl",
        model="gpt-4.1"
    )
    