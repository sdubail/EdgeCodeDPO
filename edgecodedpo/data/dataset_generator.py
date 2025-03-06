import asyncio
import json
import os
from typing import Any

import yaml
from datasets import Dataset

from edgecodedpo.clients.openai_client import OpenAIAsyncClient
from edgecodedpo.config import settings
from edgecodedpo.data.prompt_generator import (
    create_first_stage_prompt,
    create_second_stage_prompt,
    format_conversation_pair,
    generate_combinations,
)


async def process_combination(
    client: OpenAIAsyncClient,
    combination: dict[str, Any],
    system_message: str | None = None,
) -> dict[str, Any]:
    """
    Process a single combination through both stages of the pipeline.

    Args:
        client: The OpenAI client instance
        combination: The domain/task/libraries/code_form combination
        system_message: Optional system message for the OpenAI API

    Returns:
        A dictionary with the results of both stages
    """
    # Create the first stage prompt
    first_stage_prompt = create_first_stage_prompt(combination)

    # Call the OpenAI API for the first stage
    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    messages.append({"role": "user", "content": first_stage_prompt})

    print(
        f"Processing first stage for '{combination['domain']}' / '{combination['task']}' / '{combination['code_form']}'..."
    )

    first_stage_response = await client.chat_completion(
        messages=messages, json_mode=True, temperature=0.7
    )

    # Extract the content from the response
    first_content = first_stage_response["choices"][0]["message"]["content"]

    try:
        # Parse the JSON response
        first_parsed = json.loads(first_content)

        # Create the second stage prompt
        second_stage_prompt = create_second_stage_prompt(first_parsed, combination)

        # Call the OpenAI API for the second stage
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": second_stage_prompt})

        print(
            f"Processing second stage for '{combination['domain']}' / '{combination['task']}' / '{combination['code_form']}'..."
        )

        second_stage_response = await client.chat_completion(
            messages=messages,
            json_mode=True,
            temperature=0.5,  # Lower temperature for more consistent code quality
        )

        # Extract the content from the response
        second_content = second_stage_response["choices"][0]["message"]["content"]

        # Parse the JSON response
        second_parsed = json.loads(second_content)

        # Return both stages' results
        return {
            "combination": combination,
            "first_stage": {
                "prompt": first_stage_prompt,
                "response_content": first_content,
                "parsed_response": first_parsed,
            },
            "second_stage": {
                "prompt": second_stage_prompt,
                "response_content": second_content,
                "parsed_response": second_parsed,
            },
        }

    except json.JSONDecodeError as e:
        print(f"Error parsing JSON response: {e}")
        print(f"Raw response: {first_content[:200]}...")
        return {
            "combination": combination,
            "error": f"JSON parse error: {e!s}",
            "first_stage": {
                "prompt": first_stage_prompt,
                "response_content": first_content,
            },
        }
    except Exception as e:
        print(f"Unexpected error: {e}")
        return {
            "combination": combination,
            "error": f"Processing error: {e!s}",
            "first_stage": {
                "prompt": first_stage_prompt,
                "response_content": first_content,
            },
        }


async def process_batch(
    client: OpenAIAsyncClient,
    combinations: list[dict[str, Any]],
    batch_size: int = 5,
    system_message: str | None = None,
) -> list[dict[str, Any]]:
    """
    Process a batch of combinations with controlled concurrency.

    Args:
        client: The OpenAI client instance
        combinations: List of combinations to process
        batch_size: Maximum number of concurrent requests
        system_message: Optional system message for the OpenAI API

    Returns:
        List of results for each combination
    """
    semaphore = asyncio.Semaphore(batch_size)
    results = []

    async def process_with_semaphore(
        combination: list[dict[str, Any]],
    ) -> dict[str, Any]:
        async with semaphore:
            result = await process_combination(
                client=client, combination=combination, system_message=system_message
            )
            results.append(result)
            return result

    tasks = [process_with_semaphore(combo) for combo in combinations]
    await asyncio.gather(*tasks)

    return results


def convert_to_dataset_format(results: list[dict[str, Any]]) -> dict[str, list[Any]]:
    """
    Convert the API results to the HuggingFace dataset format.

    Args:
        results: List of results from the API calls

    Returns:
        Dictionary with "rejected" and "chosen" columns for HuggingFace dataset,
        along with metadata columns (domain, task, code_form)
    """
    dataset_dict: dict[str, Any] = {
        "rejected": [],
        "chosen": [],
        "domain": [],
        "task": [],
        "code_form": [],
    }

    for result in results:
        if "error" in result:
            print(f"Skipping result with error: {result['error']}")
            continue

        # Extract combination metadata
        combination = result["combination"]
        domain = combination["domain"]
        task = combination["task"]
        code_form = combination["code_form"]

        first_stage = result["first_stage"]
        second_stage = result["second_stage"]

        first_examples = first_stage["parsed_response"].get("examples", [])
        second_examples = second_stage["parsed_response"].get("improved_examples", [])
        # if len(first_examples) != len(second_examples):
        #     import pdb

        #     pdb.set_trace()
        # Process first stage examples (rejected)
        for example in first_examples:
            # Format as conversation pair
            rejected_pair = format_conversation_pair(
                prompt=example.get("prompt", ""), response=example.get("code", "")
            )
            dataset_dict["rejected"].append(rejected_pair)

        # Process second stage examples (chosen)

        for example in second_examples:
            # Get the matching first-stage prompt
            corresponding_prompt = ""
            for first_example in first_examples:
                if first_example.get("code", "") == example.get("original_code", ""):
                    corresponding_prompt = first_example.get("prompt", "")
                    break

            # Format as conversation pair
            chosen_pair = format_conversation_pair(
                prompt=corresponding_prompt, response=example.get("improved_code", "")
            )
            dataset_dict["chosen"].append(chosen_pair)

            # Add metadata for each example
            dataset_dict["domain"].append(domain)
            dataset_dict["task"].append(task)
            dataset_dict["code_form"].append(str(code_form))

    return dataset_dict


async def generate_dataset(
    config_file: str,
    output_path: str,
    num_samples: int | None = None,
    batch_size: int = 5,
    openai_model: str = "gpt-4o",
    system_message: str | None = None,
    save_intermediate: bool = True,
) -> None:
    """
    Generate and save a dataset using the OpenAI API.

    Args:
        config_file: Path to the configuration file
        output_path: Path to save the HuggingFace dataset
        num_samples: Number of combinations to sample (None for all)
        batch_size: Number of concurrent API requests
        openai_model: OpenAI model to use
        system_message: Optional system message for the API
        save_intermediate: Whether to save intermediate results
    """
    # Load the configuration
    with open(config_file) as file:
        config = yaml.safe_load(file)

    # Extract the configuration and generate combinations
    combinations = generate_combinations(config)
    print(f"Generated {len(combinations)} total combinations")

    # Sample combinations if requested
    if num_samples and num_samples < len(combinations):
        import random

        sampled_combinations = random.sample(combinations, num_samples)
        print(f"Sampled {num_samples} combinations")
    else:
        sampled_combinations = combinations

    # Initialize the OpenAI client
    client = OpenAIAsyncClient(model=openai_model, api_key=settings.OPENAI_KEY)

    # Process all combinations
    # results = await process_batch(
    #     client=client,
    #     combinations=sampled_combinations,
    #     batch_size=batch_size,
    #     system_message=system_message,
    # )

    # Save intermediate results if requested
    if save_intermediate:
        intermediate_path = os.path.join(output_path, "intermediate_results.json")
        os.makedirs(output_path, exist_ok=True)
        # with open(intermediate_path, "w", encoding="utf-8") as f:
        with open(intermediate_path, encoding="utf-8") as f:
            # json.dump(results, f, indent=2)
            results = json.load(f)
        print(f"Saved intermediate results to {intermediate_path}")

    # Convert to dataset format
    dataset_dict = convert_to_dataset_format(results)

    # Create and save the HuggingFace dataset
    dataset = Dataset.from_dict(
        {
            "rejected": dataset_dict["rejected"],
            "chosen": dataset_dict["chosen"],
            "domain": dataset_dict["domain"],
            "task": dataset_dict["task"],
            "code_form": dataset_dict["code_form"],
        }
    )

    # Save the dataset
    dataset_path = os.path.join(output_path, "dataset")
    dataset.save_to_disk(dataset_path)
    print(f"Saved HuggingFace dataset to {dataset_path}")

    # Print some statistics
    print("Dataset statistics:")
    print(f"  - Rejected examples: {len(dataset_dict['rejected'])}")
    print(f"  - Chosen examples: {len(dataset_dict['chosen'])}")
    print(f"  - Number of domains: {len(set(dataset_dict['domain']))}")
    print(f"  - Number of tasks: {len(set(dataset_dict['task']))}")
    print(f"  - Number of code forms: {len(set(dataset_dict['code_form']))}")
