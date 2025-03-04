import itertools
import json
import random
from typing import Any

import yaml


def generate_combinations(config: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Generates valid combinations of domain, task, libraries, and code form based on the
    restructured config where each task has its own allowed code forms.

    Args:
        config: The configuration dictionary with domains, tasks, libraries, and code forms

    Returns:
        List of dictionaries, each containing a valid combination
    """
    combinations = []

    for domain, domain_data in config.items():
        libraries = domain_data.get("libraries", [])
        tasks = domain_data.get("tasks", [])

        for task_entry in tasks:
            # In the new structure, each task is a dictionary with 'name' and 'code_forms'
            task_name = task_entry.get("name", "")
            code_forms = task_entry.get("code_forms", [])

            for code_form in code_forms:
                combinations.append(  # noqa: PERF401 - should be a comprehension list but let's keep it simple
                    {
                        "domain": domain,
                        "task": task_name,
                        "libraries": libraries,
                        "code_form": code_form,
                    }
                )

    return combinations


def create_first_stage_prompt(combination: dict[str, Any]) -> str:
    """
    Creates a JSON-oriented prompt template for the first stage (simple code examples).
    """
    domain = combination["domain"]
    task = combination["task"]
    libraries = combination["libraries"]
    code_form = combination["code_form"]

    libraries_str = ", ".join(libraries)

    prompt = f"""Generate 5 very different and specific Python code examples in the form of a {code_form} that address the task: "{task}" in the {domain} domain.

You can use one or more of these libraries: {libraries_str}.

For each example:
1. Create a UNIQUE and SPECIFIC use case in this domain (be very precise about the context and needs, but stay super concise.)
2. Generate Python code that directly solves the task (no type annotations, no comments, go straight to the point with beginner python level)
3. Provide a simple prompt that could be used to ask another AI model to generate exactly this code. This part very important and shouldn't be neglected.

Your response must be a valid JSON object with the following structure:
{{
  "examples": [
    {{
      "use_case": "Detailed description of the first use case",
      "code": "Your Python code here",
      "prompt": "Simple prompt to generate exactly this code"
    }},
    ... (repeat for all 5 examples)
  ]
}}

Remember that each example should correspond to a distinct use case, and the code should be targeted, practical, and directly address the task without superfluous elements.
Even if the task is already precise, find a way to propose 5 ideas that are significantly different from each other for the sake of diversity.
On the other hand, if the task is too broad/big, imagine an example that solves only a specific part of it.
"""

    return prompt


def create_second_stage_prompt(
    first_response: dict[str, Any], combination: dict[str, Any]
) -> str:
    """
    Creates a prompt for the second stage (fully typed, commented, high-quality code).

    Args:
        first_response: The parsed JSON response from the first stage
        combination: The original domain/task/libraries/code_form combination

    Returns:
        A prompt for generating improved versions of the code examples
    """
    domain = combination["domain"]
    task = combination["task"]
    libraries = combination["libraries"]
    code_form = combination["code_form"]

    # Extract the examples from the first response
    examples = first_response.get("examples", [])

    # Create a JSON-ready array of the original code examples
    code_examples_json = json.dumps(
        [{"use_case": ex["use_case"], "code": ex["code"]} for ex in examples], indent=2
    )

    prompt = f"""I have a set of Python code examples in the {domain} domain that address the task: "{task}" using the {code_form} approach. 
These examples were written to be minimal and straightforward.

Now I need you to rewrite each example with:
1. Proper type annotations following mypy standards
2. Comprehensive docstrings and comments explaining the code
3. Improved code quality and best practices where applicable
4. Any refinements that would make this production-ready, high-quality code

Here are the original examples:
{code_examples_json}

Please respond with a valid JSON object with the following structure:
{{
  "improved_examples": [
    {{
      "use_case": "The original use case description",
      "original_code": "The original code",
      "improved_code": "Your improved, typed, and commented code"
    }},
    ... (repeat for all examples)
  ]
}}

Make sure your improved code demonstrates Python expertise while maintaining the exact functionality of the original code. Focus on adding types, documentation, and improving code quality without changing the core logic or functionality."""

    return prompt


def get_random_combination(combinations: list[dict[str, Any]]) -> dict[str, Any]:
    """Gets a random combination from the list."""
    return random.choice(combinations)


def get_combination_by_index(
    combinations: list[dict[str, Any]], index: int
) -> dict[str, Any] | None:
    """Gets a specific combination by index."""
    if 0 <= index < len(combinations):
        return combinations[index]
    else:
        print(f"Index {index} out of bounds. Valid range: 0-{len(combinations)-1}")
        return None


def save_prompts_to_file(
    prompts: list[dict[str, Any]], filename: str = "generated_prompts.jsonl"
) -> None:
    """Saves the generated prompts to a JSONL file."""
    with open(filename, "w", encoding="utf-8") as f:
        for prompt_data in prompts:
            json_line = json.dumps(prompt_data, ensure_ascii=False)
            f.write(json_line + "\n")
    print(f"{len(prompts)} prompts saved to {filename}")


def generate_first_stage_prompts(
    config: dict[str, Any],
    num_samples: int | None = None,
    output_file: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Generates first-stage prompts from the configuration.

    Args:
        config: the configuration dictionary
        num_samples: Number of prompts to generate (randomly)
        output_file: Output file to save the prompts

    Returns:
        tuple: (config, combinations, prompts)
            - combinations: List of all possible combinations
            - prompts: List of dictionaries {combination, prompt_text}
    """
    if not config:
        return [], []

    # Generate all possible combinations
    combinations = generate_combinations(config)
    print(f"Generated {len(combinations)} total combinations")

    # Select the combinations to use
    selected_combinations = combinations
    if num_samples and num_samples < len(combinations):
        selected_combinations = random.sample(combinations, num_samples)
        print(f"Selected {num_samples} random combinations")

    # Generate prompts for the selected combinations
    prompts = []
    for combo in selected_combinations:
        prompt_text = create_first_stage_prompt(combo)
        prompts.append({"combination": combo, "prompt_text": prompt_text})

    # Save the prompts if an output file is specified
    if output_file and prompts:
        save_prompts_to_file(prompts, output_file)

    return combinations, prompts


def format_conversation_pair(prompt: str, response: str) -> list[dict[str, str]]:
    """
    Format a prompt-response pair as a conversation for instructional fine-tuning.

    Args:
        prompt: The prompt text
        response: The response text

    Returns:
        A list of message dictionaries in the format:
        [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}]
    """
    return [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ]


if __name__ == "__main__":
    # Example usage

    config = yaml.safe_load(open("configs/dataset.yaml"))

    combinations, prompts = generate_first_stage_prompts(config, num_samples=1)

    if prompts:
        example = prompts[0]
        combo = example["combination"]
        print("\nExample combination:")
        print(f"Domain: {combo['domain']}")
        print(f"Task: {combo['task']}")
        print(f"Libraries: {', '.join(combo['libraries'])}")
        print(f"Code form: {combo['code_form']}")

        print("\nFirst stage prompt:")
        print("-" * 50)
        print(example["prompt_text"])
        print("-" * 50)

        # Example of a first-stage response for demonstration
        example_response = {
            "examples": [
                {
                    "use_case": "Cleaning customer transaction data with missing values",
                    "code": "def clean_transactions(df):\n    df = df.dropna(subset=['transaction_id'])\n    df = df.fillna({'amount': 0, 'category': 'unknown'})\n    df = df.drop_duplicates(subset=['transaction_id'])\n    return df",
                    "prompt": "Write a pandas function to clean transaction data by removing rows with missing IDs, filling missing amounts with 0, categorizing unknowns, and removing duplicates.",
                }
            ]
        }

        # Generate second stage prompt based on the example response
        second_stage_prompt = create_second_stage_prompt(example_response, combo)

        print("\nSecond stage prompt:")
        print("-" * 50)
        print(second_stage_prompt)
        print("-" * 50)
