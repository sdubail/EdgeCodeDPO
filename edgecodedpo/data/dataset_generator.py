import asyncio
import json
import os
from typing import Any

import matplotlib.pyplot as plt  # noqa: TID253
import numpy as np
import torch
import yaml
from datasets import Dataset, load_dataset, load_from_disk
from huggingface_hub import HfApi, login
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerBase

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


async def generate_dataset_statistics(
    dataset_path: str,
    tokenizer_name_or_path: str,
    output_dir: str,
    use_gpu: bool = True,
    batch_size: int = 32,
) -> dict[str, Any]:
    """
    Generate statistics on token lengths for prompts, chosen, and rejected completions in a dataset.

    Args:
        dataset_path: Path to the dataset or HuggingFace dataset ID
        tokenizer_name_or_path: Name or path of the tokenizer to use
        output_dir: Directory to save the statistics and figures
        use_gpu: Whether to use GPU for tokenization
        batch_size: Batch size for processing

    Returns:
        Dictionary containing the statistics
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load dataset
    print(f"Loading dataset from {dataset_path}...")
    if os.path.exists(dataset_path):
        dataset = load_from_disk(dataset_path)
    else:
        # Attempt to load from HuggingFace Hub
        try:
            if ":" in dataset_path:
                repo_id, split = dataset_path.split(":", 1)
                dataset = load_dataset(repo_id, split=split)
            else:
                dataset = load_dataset(dataset_path)
                if hasattr(dataset, "keys") and "train" in dataset:
                    dataset = dataset["train"]
        except Exception as e:
            raise ValueError(f"Failed to load dataset: {e}")

    # Load tokenizer
    print(f"Loading tokenizer {tokenizer_name_or_path}...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
    print(f"Using device: {device}")

    # Prepare to collect token lengths
    prompt_lengths = []
    chosen_lengths = []
    rejected_lengths = []

    # Process dataset in batches
    for i in tqdm(range(0, len(dataset), batch_size), desc="Processing dataset"):
        batch = dataset[i : i + batch_size]

        # Process prompts
        prompt_batch = []
        for prompt in batch["prompt"]:
            # Extract prompt content from conversation format
            if (
                isinstance(prompt, list)
                and len(prompt) > 0
                and isinstance(prompt[0], dict)
            ):
                # Assumes format like [{"role": "user", "content": "..."}]
                prompt_text = prompt[0].get("content", "")
            else:
                prompt_text = str(prompt)
            prompt_batch.append(prompt_text)

        prompt_tokens = tokenizer(prompt_batch, return_tensors="pt", padding=True)
        prompt_lengths.extend(
            [sum(mask) for mask in prompt_tokens.attention_mask.tolist()]
        )

        # Process chosen completions
        chosen_batch = []
        for chosen in batch["chosen"]:
            if (
                isinstance(chosen, list)
                and len(chosen) > 0
                and isinstance(chosen[0], dict)
            ):
                chosen_text = chosen[0].get("content", "")
            else:
                chosen_text = str(chosen)
            chosen_batch.append(chosen_text)

        chosen_tokens = tokenizer(chosen_batch, return_tensors="pt", padding=True)
        chosen_lengths.extend(
            [sum(mask) for mask in chosen_tokens.attention_mask.tolist()]
        )

        # Process rejected completions
        rejected_batch = []
        for rejected in batch["rejected"]:
            if (
                isinstance(rejected, list)
                and len(rejected) > 0
                and isinstance(rejected[0], dict)
            ):
                rejected_text = rejected[0].get("content", "")
            else:
                rejected_text = str(rejected)
            rejected_batch.append(rejected_text)

        rejected_tokens = tokenizer(rejected_batch, return_tensors="pt", padding=True)
        rejected_lengths.extend(
            [sum(mask) for mask in rejected_tokens.attention_mask.tolist()]
        )

    # Calculate statistics
    stats = {
        "dataset_info": {
            "path": dataset_path,
            "size": len(dataset),
            "tokenizer": tokenizer_name_or_path,
        },
        "prompt": calculate_stats(prompt_lengths),
        "chosen": calculate_stats(chosen_lengths),
        "rejected": calculate_stats(rejected_lengths),
    }

    # Save statistics to JSON
    stats_path = os.path.join(output_dir, "token_length_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Statistics saved to {stats_path}")

    # Generate and save figures
    generate_figures(prompt_lengths, chosen_lengths, rejected_lengths, output_dir)

    return stats


def calculate_stats(values: list[int]) -> dict[str, float]:
    """Calculate statistics for a list of values."""
    values = np.array(values)
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "std": float(np.std(values)),
        "q25": float(np.percentile(values, 25)),
        "q75": float(np.percentile(values, 75)),
        "q90": float(np.percentile(values, 90)),
        "q95": float(np.percentile(values, 95)),
        "q99": float(np.percentile(values, 99)),
    }


def generate_figures(
    prompt_lengths: list[int],
    chosen_lengths: list[int],
    rejected_lengths: list[int],
    output_dir: str,
) -> None:
    """Generate and save figures showing token length distributions."""
    # Create a figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))

    # Plot histograms
    bins = np.linspace(
        0, max(max(prompt_lengths), max(chosen_lengths), max(rejected_lengths)) + 50, 50
    )

    ax1.hist(prompt_lengths, bins=bins, alpha=0.7, color="blue")
    ax1.set_title("Prompt Token Lengths")
    ax1.set_xlabel("Token Length")
    ax1.set_ylabel("Frequency")
    ax1.axvline(np.mean(prompt_lengths), color="r", linestyle="dashed", linewidth=1)
    ax1.axvline(np.median(prompt_lengths), color="g", linestyle="dashed", linewidth=1)
    ax1.grid(alpha=0.3)
    ax1.legend(["Mean", "Median", "Distribution"])

    ax2.hist(chosen_lengths, bins=bins, alpha=0.7, color="green")
    ax2.set_title("Chosen Completion Token Lengths")
    ax2.set_xlabel("Token Length")
    ax2.set_ylabel("Frequency")
    ax2.axvline(np.mean(chosen_lengths), color="r", linestyle="dashed", linewidth=1)
    ax2.axvline(np.median(chosen_lengths), color="g", linestyle="dashed", linewidth=1)
    ax2.grid(alpha=0.3)
    ax2.legend(["Mean", "Median", "Distribution"])

    ax3.hist(rejected_lengths, bins=bins, alpha=0.7, color="red")
    ax3.set_title("Rejected Completion Token Lengths")
    ax3.set_xlabel("Token Length")
    ax3.set_ylabel("Frequency")
    ax3.axvline(np.mean(rejected_lengths), color="r", linestyle="dashed", linewidth=1)
    ax3.axvline(np.median(rejected_lengths), color="g", linestyle="dashed", linewidth=1)
    ax3.grid(alpha=0.3)
    ax3.legend(["Mean", "Median", "Distribution"])

    plt.tight_layout()

    # Save the figure
    figure_path = os.path.join(output_dir, "token_length_distributions.png")
    plt.savefig(figure_path, dpi=300, bbox_inches="tight")
    print(f"Figure saved to {figure_path}")

    # Generate combined density plot for comparison
    plt.figure(figsize=(10, 6))

    # Plot kernel density estimates instead of histograms for clearer comparison
    plt.hist(
        prompt_lengths,
        bins=bins,
        alpha=0.3,
        density=True,
        color="blue",
        label="Prompts",
    )
    plt.hist(
        chosen_lengths,
        bins=bins,
        alpha=0.3,
        density=True,
        color="green",
        label="Chosen",
    )
    plt.hist(
        rejected_lengths,
        bins=bins,
        alpha=0.3,
        density=True,
        color="red",
        label="Rejected",
    )

    plt.title("Token Length Distributions Comparison")
    plt.xlabel("Token Length")
    plt.ylabel("Density")
    plt.grid(alpha=0.3)
    plt.legend()

    # Save the comparison figure
    comparison_path = os.path.join(output_dir, "token_length_comparison.png")
    plt.savefig(comparison_path, dpi=300, bbox_inches="tight")
    print(f"Comparison figure saved to {comparison_path}")
    plt.close("all")
