import asyncio
import json
import os
from typing import Any

import yaml
from datasets import Dataset, load_from_disk
from huggingface_hub import HfApi, login

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
        Dictionary with "prompt", "chosen", and "rejected" columns for HuggingFace dataset,
        along with metadata columns (domain, task, code_form)
    """
    dataset_dict: dict[str, Any] = {
        "prompt": [],
        "chosen": [],
        "rejected": [],
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

        # Process examples to create prompt-chosen-rejected format
        for example in second_examples:
            # Find corresponding first-stage example to get the prompt
            original_code = example.get("original_code", "")
            corresponding_prompt = ""

            for first_example in first_examples:
                if first_example.get("code", "") == original_code:
                    corresponding_prompt = first_example.get("prompt", "")
                    break

            if not corresponding_prompt:
                print("Warning: Could not find matching prompt for example. Skipping.")
                continue

            # Format the data in the expected conversational format
            prompt_message = [{"role": "user", "content": corresponding_prompt}]
            chosen_message = [
                {"role": "assistant", "content": example.get("improved_code", "")}
            ]
            rejected_message = [{"role": "assistant", "content": original_code}]

            # Add to dataset
            dataset_dict["prompt"].append(prompt_message)
            dataset_dict["chosen"].append(chosen_message)
            dataset_dict["rejected"].append(rejected_message)

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
    results = await process_batch(
        client=client,
        combinations=sampled_combinations,
        batch_size=batch_size,
        system_message=system_message,
    )

    # Save intermediate results if requested
    if save_intermediate:
        intermediate_path = os.path.join(output_path, "intermediate_results.json")
        os.makedirs(output_path, exist_ok=True)
        with open(intermediate_path, "w", encoding="utf-8") as f:
            # with open(intermediate_path, encoding="utf-8") as f:
            json.dump(results, f, indent=2)
            # results = json.load(f)
        print(f"Saved intermediate results to {intermediate_path}")

    # Convert to dataset format
    dataset_dict = convert_to_dataset_format(results)

    # Create and save the HuggingFace dataset
    dataset = Dataset.from_dict(
        {
            "prompt": dataset_dict["prompt"],
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


async def upload_to_huggingface(
    dataset_path: str,
    repo_id: str,
    private: bool = False,
    hf_token: str | None = None,
    fuse_datasets: bool = False,
) -> None:
    """
    Upload a saved dataset to the HuggingFace Hub.

    Args:
        dataset_path: Path to the saved HuggingFace dataset
        repo_id: ID of the repository on HuggingFace Hub (format: 'username/repo_name')
        private: Whether the repository should be private
        hf_token: HuggingFace API token (if not provided, will use the one from settings)
        fuse_datasets: Whether to look for and fuse datasets in gen_data_* directories
    """
    import glob
    import os

    from datasets import concatenate_datasets

    from edgecodedpo.config import settings

    # Use the token from settings if not provided
    token = hf_token or settings.HF_KEY
    if not token:
        raise ValueError(
            "HuggingFace API token is required. Set it in the constructor, as HF_KEY environment variable, or in .env file."
        )

    final_dataset = None

    # Check if we should fuse datasets
    if fuse_datasets:
        # Find all dataset directories matching the pattern
        base_dir = os.path.dirname(os.path.dirname(dataset_path))
        # Only match directories with numeric suffixes (gen_data_1, gen_data_2, etc.)
        pattern = os.path.join(base_dir, "gen_data_[0-9]*/dataset")

        # Find all matching dataset directories
        dataset_paths = glob.glob(pattern)

        # Sort paths to ensure consistent ordering (sorting by numeric suffix)
        dataset_paths.sort(
            key=lambda path: int(path.split("gen_data_")[1].split("/")[0])
        )

        print(f"Found {len(dataset_paths)} datasets to fuse")
        if dataset_paths:
            # Load and concatenate all datasets
            datasets = []
            for path in dataset_paths:
                print(f"Loading dataset from {path}")
                try:
                    ds = load_from_disk(path)
                    datasets.append(ds)
                    print(f"  - Loaded dataset with {len(ds)} examples")
                except Exception as e:
                    print(f"  - Error loading dataset: {e}")

            if datasets:
                # Combine the datasets
                print("Combining datasets...")
                combined_dataset = concatenate_datasets(datasets)
                print(f"Combined dataset has {len(combined_dataset)} examples")

                # Save the combined dataset
                fused_path = os.path.join(base_dir, "fused_data/dataset")
                os.makedirs(os.path.dirname(fused_path), exist_ok=True)
                combined_dataset.save_to_disk(fused_path)
                print(f"Saved combined dataset to {fused_path}")

                final_dataset = combined_dataset
            else:
                print("No valid datasets could be loaded for fusion")
        else:
            print("No datasets found for fusion")

    # If no fusion occurred or fusion failed, use the original dataset
    if final_dataset is None:
        final_dataset = load_from_disk(dataset_path)

    # Login to HuggingFace
    login(token=token)

    # Upload the dataset to HuggingFace Hub
    final_dataset.push_to_hub(
        repo_id=repo_id,
        private=private,
        token=token,
    )

    # Get the API instance for additional operations
    api = HfApi(token=token)

    # Add relevant metadata to the repository
    api.upload_file(
        path_or_fileobj=b'{"tags": ["code", "dpo", "edge-code-dpo"]}',
        path_in_repo=".tags",
        repo_id=repo_id,
        repo_type="dataset",
    )

    print(
        f"Dataset uploaded successfully to HuggingFace Hub at: https://huggingface.co/datasets/{repo_id}"
    )
