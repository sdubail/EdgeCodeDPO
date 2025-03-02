import asyncio
import json
import os
import time
from typing import Dict, List, Any, Optional
import argparse
from pathlib import Path
import yaml
from datasets import Dataset
from EdgeCodeDPO.clients.openai_client import OpenAIAsyncClient
from prompt_generator import (
    generate_combinations, 
    create_first_stage_prompt,
    create_second_stage_prompt,
    format_conversation_pair
)

async def process_combination(
    client: OpenAIAsyncClient,
    combination: Dict[str, Any],
    system_message: Optional[str] = None
) -> Dict[str, Any]:
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
    
    print(f"Processing first stage for '{combination['domain']}' / '{combination['task']}' / '{combination['code_form']}'...")
    
    first_stage_response = await client.chat_completion(
        messages=messages,
        json_mode=True,
        temperature=0.7
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
        
        print(f"Processing second stage for '{combination['domain']}' / '{combination['task']}' / '{combination['code_form']}'...")
        
        second_stage_response = await client.chat_completion(
            messages=messages,
            json_mode=True,
            temperature=0.5  # Lower temperature for more consistent code quality
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
                "parsed_response": first_parsed
            },
            "second_stage": {
                "prompt": second_stage_prompt,
                "response_content": second_content,
                "parsed_response": second_parsed
            }
        }
        
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON response: {e}")
        print(f"Raw response: {first_content[:200]}...")
        return {
            "combination": combination,
            "error": f"JSON parse error: {str(e)}",
            "first_stage": {
                "prompt": first_stage_prompt,
                "response_content": first_content
            }
        }
    except Exception as e:
        print(f"Unexpected error: {e}")
        return {
            "combination": combination,
            "error": f"Processing error: {str(e)}",
            "first_stage": {
                "prompt": first_stage_prompt,
                "response_content": first_content
            }
        }

async def process_batch(
    client: OpenAIAsyncClient,
    combinations: List[Dict[str, Any]],
    batch_size: int = 5,
    system_message: Optional[str] = None
) -> List[Dict[str, Any]]:
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
    
    async def process_with_semaphore(combination):
        async with semaphore:
            result = await process_combination(
                client=client,
                combination=combination,
                system_message=system_message
            )
            results.append(result)
            return result
    
    tasks = [process_with_semaphore(combo) for combo in combinations]
    await asyncio.gather(*tasks)
    
    return results

def convert_to_dataset_format(results: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
    """
    Convert the API results to the HuggingFace dataset format.
    
    Args:
        results: List of results from the API calls
        
    Returns:
        Dictionary with "rejected" and "chosen" columns for HuggingFace dataset
    """
    dataset_dict = {
        "rejected": [],
        "chosen": []
    }
    
    for result in results:
        if "error" in result:
            print(f"Skipping result with error: {result['error']}")
            continue
            
        first_stage = result["first_stage"]
        second_stage = result["second_stage"]
        
        # Process first stage examples (rejected)
        first_examples = first_stage["parsed_response"].get("examples", [])
        for example in first_examples:
            # Format as conversation pair
            rejected_pair = format_conversation_pair(
                prompt=example.get("prompt", ""),
                response=example.get("code", "")
            )
            dataset_dict["rejected"].append(rejected_pair)
        
        # Process second stage examples (chosen)
        second_examples = second_stage["parsed_response"].get("improved_examples", [])
        for example in second_examples:
            # Get the matching first-stage prompt
            corresponding_prompt = ""
            for first_example in first_examples:
                if first_example.get("code", "") == example.get("original_code", ""):
                    corresponding_prompt = first_example.get("prompt", "")
                    break
            
            # Format as conversation pair
            chosen_pair = format_conversation_pair(
                prompt=corresponding_prompt,
                response=example.get("improved_code", "")
            )
            dataset_dict["chosen"].append(chosen_pair)
    
    return dataset_dict

async def generate_dataset(
    config_file: str,
    output_path: str,
    num_samples: Optional[int] = None,
    batch_size: int = 5,
    openai_model: str = "gpt-4o",
    system_message: Optional[str] = None,
    save_intermediate: bool = True
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
    with open(config_file, 'r') as file:
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
    client = OpenAIAsyncClient(model=openai_model)
    
    try:
        # Process all combinations
        results = await process_batch(
            client=client,
            combinations=sampled_combinations,
            batch_size=batch_size,
            system_message=system_message
        )
        
        # Save intermediate results if requested
        if save_intermediate:
            intermediate_path = os.path.join(output_path, "intermediate_results.json")
            os.makedirs(output_path, exist_ok=True)
            with open(intermediate_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2)
            print(f"Saved intermediate results to {intermediate_path}")
        
        # Convert to dataset format
        dataset_dict = convert_to_dataset_format(results)
        
        # Create and save the HuggingFace dataset
        dataset = Dataset.from_dict({
            "rejected": dataset_dict["rejected"],
            "chosen": dataset_dict["chosen"]
        })
        
        # Save the dataset
        dataset_path = os.path.join(output_path, "dataset")
        dataset.save_to_disk(dataset_path)
        print(f"Saved HuggingFace dataset to {dataset_path}")
        
        # Print some statistics
        print(f"Dataset statistics:")
        print(f"  - Rejected examples: {len(dataset_dict['rejected'])}")
        print(f"  - Chosen examples: {len(dataset_dict['chosen'])}")
        
    finally:
        # Clean up the client
        await client.close()

def main():
    parser = argparse.ArgumentParser(description="Generate a code dataset using OpenAI API")
    parser.add_argument("--config", default="configs/dataset.yaml",type=str, help="Path to the configuration file")
    parser.add_argument("--output", default="data/gen_data", type=str, help="Path to save the dataset")
    parser.add_argument("--samples", type=int, help="Number of combinations to sample (default: all)")
    parser.add_argument("--batch-size", type=int, default=5, help="Number of concurrent API requests")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="OpenAI model to use")
    parser.add_argument("--system-message", type=str, help="System message for the API")
    parser.add_argument("--no-intermediate", action="store_true", help="Don't save intermediate results")
    
    args = parser.parse_args()
    
    # Run the generator
    asyncio.run(generate_dataset(
        config_file=args.config,
        output_path=args.output,
        num_samples=args.samples,
        batch_size=args.batch_size,
        openai_model=args.model,
        system_message=args.system_message,
        save_intermediate=not args.no_intermediate
    ))

if __name__ == "__main__":
    main()