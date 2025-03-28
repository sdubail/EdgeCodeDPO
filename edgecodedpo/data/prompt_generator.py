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
            # in the new structure each task is a dictionary with 'name' and 'code_forms'
            task_name = task_entry.get("name", "")
            code_forms = task_entry.get("code_forms", [])

            combinations.append(  # - should be a comprehension list but let's keep it simple
                {
                    "domain": domain,
                    "task": task_name,
                    "libraries": libraries,
                    "code_form": code_forms,
                }
            )

    return combinations


def create_first_stage_prompt(
    combination: dict[str, Any], is_test: bool = False, is_header: bool = False
) -> str:
    """
    Creates a JSON-oriented prompt template for the first stage (simple code examples).
    """
    domain = combination["domain"]
    task = combination["task"]
    libraries = combination["libraries"]
    code_forms = combination["code_form"]

    libraries_str = ", ".join(libraries)

    if not is_test:
        if not is_header:
            prompt = f"""Generate 3 very different and specific Python code examples, one for each code form in the list : {code_forms}, that address the task: "{task}" in the {domain} domain.

You can use one or more of these libraries: {libraries_str}.

For each example:
1. Create a UNIQUE and SPECIFIC use case in this domain addressing the task e study (be very precise about the context and needs, but stay super concise.)
2. Generate Python code that directly solves the task (no type annotations, no comments, no docstrings, go straight to the point with beginner python level)
3. Provide a simple prompt that could be used by a human to ask another AI model to generate the code solving the use case. This part very important and shouldn't be neglected.

Your response must be a valid JSON object with the following structure:
{{
  "examples": [
    {{
      "use_case": "Detailed description of the first use case",
      "code": "Your Python code here",
      "prompt": "Simple prompt to generate this code"
    }},
    ... (repeat for all 3 examples)
  ]
}}

Remember that each example should correspond to a distinct use case for a distinct code form, and the code should be targeted, practical, and directly address the task without superfluous elements.
Even if the task is already precise, find a way to propose 3 ideas that are significantly different from each other for the sake of diversity.
On the other hand, if the task is too broad/big, imagine an example that solves only a specific part of it.
"""
        else:
            prompt = f"""Generate 3 very different and specific Python function headers, one for each code form in the list : {code_forms}, to address the task: "{task}" in the {domain} domain.

You can use one or more of these libraries: {libraries_str}.

For each example:
1. Create a SPECIFIC and UNIQUE use case in this domain (be precise about the context and needs, but stay concise).
2. Generate ONLY the function header including the function name and parameters - NO implementation code, besides the necessary imports. For classes, we also want the init function header inside. 
Make the header BASIC: absolutely no type annotations, no docstrings.
3. Provide a simple prompt that could be used by a human to ask another AI model to generate the function header for the use case. 
The prompt part very important and shouldn't be neglected. The prompt should mention that ONLY the function header is requested, and also briefly mention the use case like a human would.
It shouldn't specify too much the details about the header parameters and/or function name (which a human wouldn't do), and stay elusive, for instance : "can you code me the header of a function (or any code form) for <use_case>". This is just an example, please do your own prompts and have variations ! 

Your response must be a valid JSON object with the following structure:
{{
  "headers": [
    {{
    "use_case": "Detailed description of the use case",
    "header": "imports, and simple header, like def function_name(param1, param2):"
    "prompt": "Simple prompt to generate this code"
    }}
    ... (repeat for all 3 headers)
  ]
}}
"""
    else:
        if not is_header:
            prompt = f"""Generate a Python code example that uses the {code_forms} approach to address the task: "{task}" in the {domain} domain.

You can use one or more of these libraries: {libraries_str}.

Create a SPECIFIC and UNIQUE use case in this domain (be precise about the context and needs, but stay concise).
Generate Python code that directly solves the task (no type annotations, no comments, go straight to the point with beginner python level).

Your response must be a valid JSON object with the following structure:
{{
  "use_case": "Detailed description of the use case",
  "code": "Your Python code here"
}}

Remember to make the code targeted, practical, and directly address the task without superfluous elements.
"""
        else:
            raise ValueError("No test dataset with header only mode.")
    return prompt


def create_second_stage_prompt(
    first_response: dict[str, Any],
    combination: dict[str, Any],
    is_test: bool = False,
    is_header: bool = False,
) -> str:
    """
    Creates a prompt for the second stage (fully typed, commented, high-quality code).

    Args:
        first_response: The parsed JSON response from the first stage
        combination: The original domain/task/libraries/code_form combination
        is_test : is the dataset generated for test or for train

    Returns:
        A prompt for generating improved versions of the code examples
    """
    domain = combination["domain"]
    task = combination["task"]
    libraries = combination["libraries"]
    code_form = combination["code_form"]

    if not is_test:
        if not is_header:
            examples = first_response.get("examples", [])

            code_examples_json = json.dumps(
                [{"use_case": ex["use_case"], "code": ex["code"]} for ex in examples],
                indent=2,
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

Remember that you can use the following libraries : {libraries}.

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
        else:
            headers = first_response.get("headers", [])

            headers_json = json.dumps(
                [
                    {"use_case": ex["use_case"], "header": ex["header"]}
                    for ex in headers
                ],
                indent=2,
            )

            prompt = f"""I have a set of Python function headers in the {domain} domain that address the task: "{task}" using the {code_form} approach. 
These headers were written to be minimal and straightforward. For classes we also have the header of the init method.

Now I need you to rewrite each header with:
1. Proper mypy type annotations for all parameters and return values
2. Comprehensive docstrings explaining:
   - The function's purpose
   - Each parameter (with types)
   - Return value (with type)
   - Any raised exceptions
3. Improved naming and parameter structure following Python best practices

Here are the original headers:
{headers_json}

Please respond with a valid JSON object with the following structure:
{{
  "improved_headers": [
    {{
      "use_case": "The original use case description",
      "original_header": "The original header",
      "improved_header": "Your improved, typed, and documented header"
    }},
    ... (repeat for all headers)
  ]
}}

Make sure your improved headers demonstrate Python expertise with excellent type annotations and documentation. Do not include implementation code, focus only on creating exemplary function or class headers. Do not forget docstrings for classes too !"""

    else:
        if not is_header:
            use_case = first_response.get("use_case", "")
            code = first_response.get("code", "")

            code_example_json = json.dumps(
                {"use_case": use_case, "code": code}, indent=2
            )

            prompt = f"""I have a Python code example in the {domain} domain that addresses the task: "{task}" using the {code_form} approach. 
This example was written to be minimal and straightforward.

Now I need you to rewrite the example with:
1. Proper type annotations following mypy standards
2. Comprehensive docstrings and comments explaining the code
3. Improved code quality and best practices where applicable
4. Any refinements that would make this production-ready, high-quality code

Additionally, generate THREE different prompts that could be used to request this code:
1. A default prompt that simply asks for code to solve the use case
2. A prompt that asks for code to solve the use case and specifies the code form (e.g., "function", "class") to use
3. A prompt that asks for code to solve the use case and specifies both the code form AND the explicit input/output types

Here is the original example:
{code_example_json}

Please respond with a valid JSON object with the following structure:
{{
"use_case": "The original use case description",
"original_code": "The original code",
"improved_code": "Your improved, typed, and commented code",
"prompt_default": "A simple prompt to generate code for this use case",
"prompt_code_form": "A prompt that specifies the code form to use",
"prompt_code_form_types": "A prompt that specifies code form and hints at input/output types"
}}

Make sure your improved code demonstrates Python expertise while maintaining the exact functionality of the original code. Focus on adding types, documentation, and improving code quality without changing the core logic or functionality."""
        else:
            raise ValueError("No test dataset with header only mode.")
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
