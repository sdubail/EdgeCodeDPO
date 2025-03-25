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
from transformers import AutoTokenizer

from edgecodedpo.clients.openai_client import OpenAIAsyncClient
from edgecodedpo.config import settings
from edgecodedpo.data.prompt_generator import (
    create_first_stage_prompt,
    create_second_stage_prompt,
    generate_combinations,
)


async def process_combination(
    client: OpenAIAsyncClient,
    combination: dict[str, Any],
    system_message: str | None = None,
    is_test: bool = False,
    is_header: bool = False,
) -> dict[str, Any]:
    """
    Process a single combination through both stages of the pipeline.

    Args:
        client: The OpenAI client instance
        combination: The domain/task/libraries/code_form combination
        system_message: Optional system message for the OpenAI API
        is_test: is the dataset generated for test or train purposes
        is_header: is the dataset in headers only or full code mode

    Returns:
        A dictionary with the results of both stages
    """
    # first stage prompt
    first_stage_prompt = create_first_stage_prompt(
        combination, is_test=is_test, is_header=is_header
    )

    # call the OpenAI API for the first stage
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

    # extraction
    first_content = first_stage_response["choices"][0]["message"]["content"]

    try:
        # parsing
        first_parsed = json.loads(first_content)

        # create the second stage prompt
        second_stage_prompt = create_second_stage_prompt(
            first_parsed, combination, is_test=is_test, is_header=is_header
        )

        # call API for the second stage
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
            temperature=0.5,  # lower temperature for more consistent code quality
        )

        second_content = second_stage_response["choices"][0]["message"]["content"]

        second_parsed = json.loads(second_content)

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
    is_test: bool = False,
    is_header: bool = False,
) -> list[dict[str, Any]]:
    """
    Process a batch of combinations with controlled concurrency.

    Args:
        client: The OpenAI client instance
        combinations: List of combinations to process
        batch_size: Maximum number of concurrent requests
        system_message: Optional system message for the OpenAI API
        is_test: Is the dataset generated for test or train purposes
        is_header: Is the dataset in headers only or full code mode

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
                client=client,
                combination=combination,
                system_message=system_message,
                is_test=is_test,
                is_header=is_header,
            )
            results.append(result)
            return result

    tasks = [process_with_semaphore(combo) for combo in combinations]
    await asyncio.gather(*tasks)

    return results


def convert_to_dataset_format(
    results: list[dict[str, Any]], is_header: bool = False
) -> dict[str, list[Any]]:
    """
    Convert the API results to the HuggingFace dataset format.

    Args:
        results: List of results from the API calls
        is_header: Is the dataset in headers only or full code mode

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

        # extract combination metadata
        combination = result["combination"]
        domain = combination["domain"]
        task = combination["task"]
        code_form = combination["code_form"]

        first_stage = result["first_stage"]
        second_stage = result["second_stage"]

        if not is_header:
            original_main_key = "examples"
            improved_main_key = "improved_examples"
            original_key = "original_code"
            improved_key = "improved_code"
            base_key = "code"
        else:
            original_main_key = "headers"
            improved_main_key = "improved_headers"
            original_key = "original_header"
            improved_key = "improved_header"
            base_key = "header"

        first_examples = first_stage["parsed_response"].get(original_main_key, [])
        second_examples = second_stage["parsed_response"].get(improved_main_key, [])

        # process examples to create prompt-chosen-rejected format
        for example in second_examples:
            original_code = example.get(original_key, "")
            corresponding_prompt = ""

            for first_example in first_examples:
                if first_example.get(base_key, "") == original_code:
                    corresponding_prompt = first_example.get("prompt", "")
                    break

            if not corresponding_prompt:
                print("Warning: Could not find matching prompt for example. Skipping.")
                continue

            # format the data in the expected conversational format
            prompt_message = [{"role": "user", "content": corresponding_prompt}]
            chosen_message = [
                {"role": "assistant", "content": example.get(improved_key, "")}
            ]
            rejected_message = [{"role": "assistant", "content": original_code}]

            # add to dataset
            dataset_dict["prompt"].append(prompt_message)
            dataset_dict["chosen"].append(chosen_message)
            dataset_dict["rejected"].append(rejected_message)

            # add metadata for each example
            dataset_dict["domain"].append(domain)
            dataset_dict["task"].append(task)
            dataset_dict["code_form"].append(str(code_form))

    return dataset_dict


def convert_to_test_dataset_format(
    results: list[dict[str, Any]],
) -> dict[str, list[Any]]:
    """
    Convert the API results to the HuggingFace dataset format for testing.
    Creates multiple rows for each example, one for each prompt type.

    Args:
        results: List of results from the API calls

    Returns:
        Dictionary ready for creating a HuggingFace dataset
    """
    dataset_dict: dict[str, list[Any]] = {
        "prompt": [],
        "chosen": [],
        "rejected": [],
        "domain": [],
        "task": [],
        "code_form": [],
        "prompt_type": [],
    }

    for result in results:
        if "error" in result:
            print(f"Skipping result with error: {result['error']}")
            continue

        combination = result["combination"]
        domain = combination["domain"]
        task = combination["task"]
        code_form = combination["code_form"][0]  # just one code form /task

        second_stage = result["second_stage"]
        processed_response = second_stage["parsed_response"]

        # get improved and orig
        improved_code = processed_response.get("improved_code", "")
        original_code = processed_response.get("original_code", "")

        # for each prompt type create a separate dataset entry
        prompt_types = [
            ("prompt_default", "default"),
            ("prompt_code_form", "code_form"),
            ("prompt_code_form_types", "code_form_types"),
        ]

        for prompt_field, prompt_type in prompt_types:
            prompt_text = processed_response.get(prompt_field, "")
            if not prompt_text:
                continue

            prompt_message = [{"role": "user", "content": prompt_text}]
            chosen_message = [{"role": "assistant", "content": improved_code}]
            rejected_message = [{"role": "assistant", "content": original_code}]

            dataset_dict["prompt"].append(prompt_message)
            dataset_dict["chosen"].append(chosen_message)
            dataset_dict["rejected"].append(rejected_message)

            dataset_dict["domain"].append(domain)
            dataset_dict["task"].append(task)
            dataset_dict["code_form"].append(code_form)
            dataset_dict["prompt_type"].append(prompt_type)

    return dataset_dict


async def generate_dataset(
    config_file: str,
    output_path: str,
    is_test: bool = False,
    is_header: bool = False,
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
        is_test : Is the dataset on test mode or train mode
        is_header : Is the dataset on headers only mode or full code mode
        num_samples: Number of combinations to sample (None for all)
        batch_size: Number of concurrent API requests
        openai_model: OpenAI model to use
        system_message: Optional system message for the API
        save_intermediate: Whether to save intermediate results
    """
    if is_test:
        output_path += "_test"

    with open(config_file) as file:
        config = yaml.safe_load(file)

    combinations = generate_combinations(config)
    print(f"Generated {len(combinations)} total combinations")

    if num_samples and num_samples < len(combinations):
        import random

        sampled_combinations = random.sample(combinations, num_samples)
        print(f"Sampled {num_samples} combinations")
    else:
        sampled_combinations = combinations

    client = OpenAIAsyncClient(model=openai_model, api_key=settings.OPENAI_KEY)

    # process all combinations
    results = await process_batch(
        client=client,
        combinations=sampled_combinations,
        batch_size=batch_size,
        system_message=system_message,
        is_test=is_test,
        is_header=is_header,
    )

    if save_intermediate:
        intermediate_path = os.path.join(output_path, "intermediate_results.json")
        os.makedirs(output_path, exist_ok=True)
        with open(intermediate_path, "w", encoding="utf-8") as f:
            # with open(intermediate_path, encoding="utf-8") as f:
            json.dump(results, f, indent=2)
            # results = json.load(f)
        print(f"Saved intermediate results to {intermediate_path}")

    if not is_test:
        dataset_dict = convert_to_dataset_format(results, is_header=is_header)
    else:
        dataset_dict = convert_to_test_dataset_format(results)

    dataset = Dataset.from_dict(dataset_dict)

    # aave the dataset
    dataset_path = os.path.join(output_path, "dataset")
    dataset.save_to_disk(dataset_path)
    print(f"Saved HuggingFace dataset to {dataset_path}")

    # some statistics
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

    token = hf_token or settings.HF_KEY
    if not token:
        raise ValueError(
            "HuggingFace API token is required. Set it in the constructor, as HF_KEY environment variable, or in .env file."
        )

    final_dataset = None

    if fuse_datasets:
        base_dir = os.path.dirname(os.path.dirname(dataset_path))
        pattern = os.path.join(base_dir, "gen_data_[0-9]*/dataset")

        dataset_paths = glob.glob(pattern)

        dataset_paths.sort(
            key=lambda path: int(path.split("gen_data_")[1].split("/")[0])
        )

        print(f"Found {len(dataset_paths)} datasets to fuse")
        if dataset_paths:
            # load and concatenate all datasets
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
                # combine the datasets
                print("Combining datasets...")
                combined_dataset = concatenate_datasets(datasets)
                print(f"Combined dataset has {len(combined_dataset)} examples")

                # save the combined dataset
                fused_path = os.path.join(base_dir, "fused_data/dataset")
                os.makedirs(os.path.dirname(fused_path), exist_ok=True)
                combined_dataset.save_to_disk(fused_path)
                print(f"Saved combined dataset to {fused_path}")

                final_dataset = combined_dataset
            else:
                print("No valid datasets could be loaded for fusion")
        else:
            print("No datasets found for fusion")

    if final_dataset is None:
        final_dataset = load_from_disk(dataset_path)

    login(token=token)

    final_dataset.push_to_hub(
        repo_id=repo_id,
        private=private,
        token=token,
    )

    api = HfApi(token=token)

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
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading dataset from {dataset_path}...")
    if os.path.exists(dataset_path):
        dataset = load_from_disk(dataset_path)
    else:
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

    print(f"Loading tokenizer {tokenizer_name_or_path}...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)

    device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
    print(f"Using device: {device}")

    prompt_lengths = []
    chosen_lengths = []
    rejected_lengths = []

    # process dataset in batches
    for i in tqdm(range(0, len(dataset), batch_size), desc="Processing dataset"):
        batch = dataset[i : i + batch_size]

        # Process prompts
        prompt_batch = []
        for prompt in batch["prompt"]:
            if (
                isinstance(prompt, list)
                and len(prompt) > 0
                and isinstance(prompt[0], dict)
            ):
                # assumes format like [{"role": "user", "content": "..."}]
                prompt_text = prompt[0].get("content", "")
            else:
                prompt_text = str(prompt)
            prompt_batch.append(prompt_text)

        prompt_tokens = tokenizer(prompt_batch, return_tensors="pt", padding=True)
        prompt_lengths.extend(
            [sum(mask) for mask in prompt_tokens.attention_mask.tolist()]
        )

        # process chosen completions
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

        # process rejected completions
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

    stats_path = os.path.join(output_dir, "token_length_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Statistics saved to {stats_path}")

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
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))

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

    figure_path = os.path.join(output_dir, "token_length_distributions.png")
    plt.savefig(figure_path, dpi=300, bbox_inches="tight")
    print(f"Figure saved to {figure_path}")

    plt.figure(figsize=(10, 6))

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
