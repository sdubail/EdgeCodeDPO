"""
DPO training module for EdgeCodeDPO.

This module provides functionality to train language models using Direct Preference Optimization (DPO)
as described in the paper "Direct Preference Optimization: Your Language Model is Secretly a Reward Model"
by Rafailov et al.
"""

import json
import os
from typing import Any

import numpy as np
import torch
from datasets import Dataset, load_dataset, load_from_disk
from peft import AutoPeftModelForCausalLM, LoraConfig, get_peft_model
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import DPOConfig, DPOTrainer

from edgecodedpo.config import settings
from edgecodedpo.training.eval_metrics import (
    calculate_code_similarity,
    evaluate_code_quality,
    execute_code,
)
from edgecodedpo.training.visualization import create_visualizations
from edgecodedpo.utils.generated_code_parsing import (
    assemble_code_blocks,
    extract_code_blocks,
    preprocess_code_blocks,
)


def load_model_and_tokenizer(
    model_name_or_path: str,
    dpo_config: DPOConfig,
    quantization_config: dict[str, Any] | None = None,
    lora_config: dict[str, Any] | None = None,
    is_already_lora: bool = False,
) -> tuple:
    """
    Load the model and tokenizer for DPO training.

    Args:
        model_name_or_path: Name or path of the base model or fine-tuned model
        dpo_config: DPO configuration
        quantization_config: Configuration for quantization (BitsAndBytes)
        lora_config: Configuration for LoRA
        is_already_lora: if the model is already a lora, load adapter twice

    Returns:
        tuple: (model, tokenizer)
    """
    # Set up quantization if specified
    bnb_config = None
    if quantization_config:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=quantization_config.get("load_in_4bit", False),
            load_in_8bit=quantization_config.get("load_in_8bit", False),
            llm_int8_threshold=quantization_config.get("llm_int8_threshold", 6.0),
            llm_int8_has_fp16_weight=quantization_config.get(
                "llm_int8_has_fp16_weight", False
            ),
            bnb_4bit_compute_dtype=getattr(
                torch, quantization_config.get("bnb_4bit_compute_dtype", "float16")
            ),
            bnb_4bit_use_double_quant=quantization_config.get(
                "bnb_4bit_use_double_quant", True
            ),
            bnb_4bit_quant_type=quantization_config.get("bnb_4bit_quant_type", "nf4"),
        )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        padding_side="right",
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model with quantization if specified
    model_kwargs = {
        "trust_remote_code": True,
        "device_map": "auto",
    }

    if bnb_config:
        model_kwargs["quantization_config"] = bnb_config

    if not is_already_lora:
        # Load model
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)

        model.config.use_cache = False

        # Apply LoRA if specified
        if lora_config:
            peft_config = LoraConfig(
                r=lora_config.get("r", 16),
                lora_alpha=lora_config.get("lora_alpha", 16),
                lora_dropout=lora_config.get("lora_dropout", 0.05),
                target_modules=lora_config.get("target_modules", None),
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, peft_config)

    else:
        model = AutoPeftModelForCausalLM.from_pretrained(
            model_name_or_path, **model_kwargs
        )
        # Load the adapter a second time, with a different name, which will be our reference model.
        model.load_adapter(model_name_or_path, adapter_name="reference")

    return model, tokenizer


def preprocess_dataset(
    dataset_path: str,
    tokenizer: Any,
    max_length: int = 1024,
    num_proc: int = 4,
) -> Dataset:
    """
    Preprocess the dataset for DPO training.
    Supports loading from local path or directly from the HuggingFace Hub.

    Args:
        dataset_path: Path to the dataset or HuggingFace Hub dataset ID
        tokenizer: Tokenizer to use for preprocessing
        max_length: Maximum sequence length
        num_proc: Number of processes to use for preprocessing

    Returns:
        Dataset: Preprocessed dataset
    """
    # Check if the dataset_path is a local path
    if os.path.exists(dataset_path):
        # Load dataset from local path
        print(f"Loading dataset from local path: {dataset_path}")
        dataset = load_from_disk(dataset_path)
    else:
        # Attempt to load from HuggingFace Hub
        print(f"Attempting to load dataset from HuggingFace Hub: {dataset_path}")
        try:
            # Check if a specific split is specified (e.g., "dataset_name:train")
            if ":" in dataset_path:
                repo_id, split = dataset_path.split(":", 1)
                dataset = load_dataset(repo_id, split=split)
            else:
                # Try loading the default split
                dataset = load_dataset(dataset_path)
                # If dataset is a DatasetDict with multiple splits, prefer 'train'
                if hasattr(dataset, "keys") and "train" in dataset:
                    dataset = dataset["train"]
        except Exception as e:
            raise ValueError(
                f"Failed to load dataset from HuggingFace Hub: {dataset_path}. Error: {e!s}"
            )

    # Make sure it has the expected format with chosen and rejected samples
    required_columns = ["chosen", "rejected"]
    if not all(col in dataset.column_names for col in required_columns):
        raise ValueError(
            f"Dataset must contain columns: {required_columns}. Found: {dataset.column_names}"
        )

    # If the dataset is already preprocessed with the right format, return it
    if "chosen" in dataset.column_names and isinstance(dataset["chosen"][0], list):
        print("Dataset already in the expected format. Skipping preprocessing.")
        return dataset

    # Apply tokenization/formatting if needed
    # Here you would implement any specific preprocessing steps needed for your dataset

    return dataset


def train_dpo(
    model_name_or_path: str,
    dataset_path: str,
    output_dir: str,
    dpo_config: dict[str, Any] | None = None,
    quantization_config: dict[str, Any] | None = None,
    lora_config: dict[str, Any] | None = None,
    is_already_lora: bool = False,
    push_to_hub: bool = False,
    hub_model_id: str | None = None,
) -> None:
    """
    Train a model using DPO.

    Args:
        model_name_or_path: Name or path of the base model or fine-tuned model
        dataset_path: Path to the dataset
        output_dir: Directory to save the trained model
        dpo_config: DPO configuration
        quantization_config: Configuration for quantization
        lora_config: Configuration for LoRA
        is_already_lora: is the model already a lora, in that case load adapter twice
        push_to_hub: Whether to push the model to HuggingFace Hub
        hub_model_id: ID for the HuggingFace repository
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    model_adapter_name = "train" if is_already_lora else None
    ref_adapter_name = "reference" if is_already_lora else None

    # Set up DPO configuration
    training_args = DPOConfig(
        output_dir=output_dir,
        num_train_epochs=int(dpo_config.get("num_train_epochs", 3)),
        per_device_train_batch_size=int(
            dpo_config.get("per_device_train_batch_size", 4)
        ),
        per_device_eval_batch_size=int(dpo_config.get("per_device_eval_batch_size", 4)),
        gradient_accumulation_steps=int(
            dpo_config.get("gradient_accumulation_steps", 1)
        ),
        learning_rate=float(dpo_config.get("learning_rate", 1e-6)),
        lr_scheduler_type=dpo_config.get("lr_scheduler_type", "cosine"),
        warmup_ratio=float(dpo_config.get("warmup_ratio", 0.1)),
        logging_steps=int(dpo_config.get("logging_steps", 10)),
        save_strategy=dpo_config.get("save_strategy", "steps"),
        save_steps=int(dpo_config.get("save_steps", 100)),
        evaluation_strategy=dpo_config.get("evaluation_strategy", "steps"),
        eval_steps=int(dpo_config.get("eval_steps", 100)),
        beta=float(dpo_config.get("beta", 0.1)),
        loss_type=dpo_config.get("loss_type", "sigmoid"),
        max_length=int(dpo_config.get("max_length", 1024)),
        max_prompt_length=int(dpo_config.get("max_prompt_length", 512)),
        bf16=dpo_config.get("bf16", True),
        fp16=dpo_config.get("fp16", False),
        precompute_ref_log_probs=dpo_config.get("precompute_ref_log_probs", False),
        logging_dir=os.path.join(output_dir, "logs"),
        model_adapter_name=model_adapter_name,
        ref_adapter_name=ref_adapter_name,
        # report_to=dpo_config.get("report_to", "tensorboard"),
        push_to_hub=push_to_hub,
        hub_model_id=hub_model_id,
        hub_token=settings.HF_KEY if push_to_hub else None,
    )

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(
        model_name_or_path=model_name_or_path,
        dpo_config=training_args,
        quantization_config=quantization_config,
        lora_config=lora_config,
        is_already_lora=is_already_lora,
    )

    # Load and preprocess dataset
    dataset = preprocess_dataset(
        dataset_path=dataset_path,
        tokenizer=tokenizer,
        max_length=dpo_config.get("max_length", 1024),
        num_proc=dpo_config.get("dataset_num_proc", 4),
    )

    # Split dataset into train/eval if not already done
    if "train" not in dataset and "test" not in dataset:
        dataset = dataset.train_test_split(test_size=dpo_config.get("eval_split", 0.1))
        train_dataset = dataset["train"]
        eval_dataset = dataset["test"]
    else:
        train_dataset = dataset.get("train", dataset)
        eval_dataset = dataset.get("test", None)

    # Initialize DPO trainer
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    # Train model
    train_result = trainer.train()

    # Save the trained model
    trainer.save_model(output_dir)

    # Log and save metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    # Generate visualization plots from TensorBoard logs
    log_dir = os.path.join(output_dir, "logs")
    viz_dir = os.path.join(output_dir, "visualizations")
    try:
        print("Generating metric visualizations...")
        create_visualizations(log_dir=log_dir, output_dir=viz_dir)
        print(f"Visualizations saved to {viz_dir}")
    except Exception as e:
        print(f"Warning: Could not generate visualizations: {e}")

    # Push to hub if requested
    if push_to_hub and hub_model_id:
        trainer.push_to_hub()
        print(
            f"Model pushed to HuggingFace Hub at: https://huggingface.co/{hub_model_id}"
        )

    print(f"Training completed. Model saved to: {output_dir}")


def load_and_evaluate_model(
    model_path: str,
    dataset_path: str,
    output_dir: str,
    is_test: bool = False,
    max_length: int = 1024,
    num_examples: int = 10,
    batch_size: int = 4,
    add_prompt_boost: bool = False,
) -> None:
    """
    Load a trained model and evaluate it on the dataset.

    Args:
        model_path: Path to the trained model
        dataset_path: Path to the dataset
        output_dir: Directory to save the evaluation results
        is_test: Whether the dataset is a test dataset with prompt types
        max_length: Maximum sequence length
        num_examples: Number of examples to evaluate
        batch_size: Number of examples to process in a single batch
        add_prompt_boost: Whether to add prompt engineering to the base prompts
    """
    # Load model and tokenizer
    try:
        # Try loading as a PEFT model first
        model = AutoPeftModelForCausalLM.from_pretrained(
            model_path, device_map="auto", trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    except Exception:
        # Fall back to standard model loading
        model = AutoModelForCausalLM.from_pretrained(
            model_path, device_map="auto", trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Load dataset
    # Check if the dataset_path is a local path
    if os.path.exists(dataset_path):
        # Load dataset from local path
        print(f"Loading dataset from local path: {dataset_path}")
        dataset = load_from_disk(dataset_path)
    else:
        # Attempt to load from HuggingFace Hub
        print(f"Attempting to load dataset from HuggingFace Hub: {dataset_path}")
        try:
            # Check if a specific split is specified (e.g., "dataset_name:train")
            if ":" in dataset_path:
                repo_id, split = dataset_path.split(":", 1)
                dataset = load_dataset(repo_id, split=split)
            else:
                # Try loading the default split
                dataset = load_dataset(dataset_path)
        except Exception as e:
            raise ValueError(
                f"Failed to load dataset from HuggingFace Hub: {dataset_path}. Error: {e!s}"
            )

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Select evaluation examples
    if "test" in dataset:
        eval_dataset = dataset["test"].select(
            range(min(num_examples, dataset["test"].num_rows))
        )
    else:
        eval_dataset = dataset["train"].select(
            range(min(num_examples, dataset["train"].num_rows))
        )

    # Initialize overall metrics dictionary
    metrics = {
        "type_annotation_coverage": [],
        "comment_density": [],
        "docstring_coverage": [],
        "code_complexity": [],
        "pep8_compliance": [],
        "execution_success_rate": 0,
        "similarity_to_chosen": [],
        "similarity_to_rejected": [],
    }

    # Initialize metrics by prompt type dictionary if this is a test dataset
    prompt_types_metrics = {}
    prompt_types_counts = {}

    # Generate predictions and evaluate
    results = []
    fails = []
    # Process dataset in batches
    for batch_start in tqdm(range(0, len(eval_dataset), batch_size)):
        batch_end = min(batch_start + batch_size, len(eval_dataset))
        batch = eval_dataset[batch_start:batch_end]

        # Get prompts and prompt types from the batch
        # Note: batch is a dict with column names as keys and lists of values
        batch_prompts = batch.get("prompt", [])
        batch_prompt_types = batch.get("prompt_type", [None] * len(batch_prompts))

        # Process each example in the batch
        valid_inputs = []
        valid_indices = []

        for i in range(len(batch_prompts)):
            prompt = batch_prompts[i] if i < len(batch_prompts) else None
            prompt_type = (
                batch_prompt_types[i] if i < len(batch_prompt_types) else "default"
            )

            if not is_test and not prompt_type:
                prompt_type = "default"

            # Initialize metrics for this prompt type if we haven't seen it before
            if prompt_type and prompt_type not in prompt_types_metrics:
                prompt_types_metrics[prompt_type] = {
                    "type_annotation_coverage": [],
                    "comment_density": [],
                    "docstring_coverage": [],
                    "code_complexity": [],
                    "pep8_compliance": [],
                    "execution_success_rate": 0,
                    "similarity_to_chosen": [],
                    "similarity_to_rejected": [],
                }
                prompt_types_counts[prompt_type] = 0

            if prompt_type:
                prompt_types_counts[prompt_type] += 1

            # Skip examples without prompts
            if not prompt:
                continue

            # Extract text prompt from the conversation format
            if (
                isinstance(prompt, list)
                and len(prompt) > 0
                and isinstance(prompt[0], dict)
            ):
                text_prompt = prompt[0].get("content", "")
            else:
                text_prompt = str(prompt)
            if add_prompt_boost:
                text_prompt += "Use docstrings and mypy typing."
            valid_inputs.append(text_prompt)
            valid_indices.append(i)

        # Skip batch if no valid inputs
        if not valid_inputs:
            continue

        # Tokenize all valid prompts in the batch
        batch_inputs = tokenizer(valid_inputs, return_tensors="pt", padding=True).to(
            model.device
        )
        print(f"Input length:{len(batch_inputs.input_ids)}")
        # Generate completions for the batch
        batch_outputs = model.generate(
            input_ids=batch_inputs.input_ids,
            attention_mask=batch_inputs.attention_mask,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.7,
        )
        print(f"Output length:{len(batch_outputs)}")
        # Process each generated output
        for j, idx in enumerate(valid_indices):
            prompt = batch_prompts[idx]
            prompt_type = (
                batch_prompt_types[idx] if idx < len(batch_prompt_types) else "default"
            )
            chosen = (
                batch.get("chosen", [])[idx]
                if idx < len(batch.get("chosen", []))
                else None
            )
            rejected = (
                batch.get("rejected", [])[idx]
                if idx < len(batch.get("rejected", []))
                else None
            )

            # Decode the generated output (each output is for a single example)
            output_ids = batch_outputs[j]
            generated_code = tokenizer.decode(output_ids, skip_special_tokens=True)

            code_blocks = extract_code_blocks(generated_code)
            full_script = assemble_code_blocks(code_blocks)
            preprocess_blocks = preprocess_code_blocks(code_blocks)
            if len(preprocess_blocks) == 0:
                fails.append(generated_code)
                print("No code found, skipping.")
                continue

            block_metrics = {
                "type_annotation_coverage": [],
                "comment_density": [],
                "docstring_coverage": [],
                "code_complexity": [],
                "pep8_compliance": [],
            }
            for code_block in preprocess_blocks:
                # Evaluate code quality
                code_metrics = evaluate_code_quality(code_block)
                for key, value in code_metrics.items():
                    block_metrics[key].append(value)

            # Evaluate functional correctness
            execution_result = execute_code(full_script)
            if execution_result["success"]:
                metrics["execution_success_rate"] += 1
                if prompt_type:
                    prompt_types_metrics[prompt_type]["execution_success_rate"] += 1

            # Averaging metrics across code blocks
            for metric_name, metric_value in block_metrics.items():
                if isinstance(metric_value, list):
                    filtered_values = list(
                        filter(lambda x: x is not None, metric_value)
                    )
                    if filtered_values:
                        avg_value = np.mean(filtered_values)
                        metrics[metric_name].append(avg_value)
                        if prompt_type:
                            prompt_types_metrics[prompt_type][metric_name].append(
                                avg_value
                            )
                elif isinstance(metric_value, int):
                    avg_value = metric_value / len(preprocess_blocks)
                    metrics[metric_name] = avg_value
                    if prompt_type:
                        prompt_types_metrics[prompt_type][metric_name] = avg_value

            # Compare with chosen code
            if chosen:
                text_chosen_code = chosen[0]["content"]
                similarity_to_chosen = calculate_code_similarity(
                    generated_code, text_chosen_code
                )
                metrics["similarity_to_chosen"].append(similarity_to_chosen)
                if prompt_type:
                    prompt_types_metrics[prompt_type]["similarity_to_chosen"].append(
                        similarity_to_chosen
                    )
            else:
                similarity_to_chosen = None

            # Compare with rejected code
            if rejected:
                text_rejected_code = rejected[0]["content"]
                similarity_to_rejected = calculate_code_similarity(
                    generated_code, text_rejected_code
                )
                metrics["similarity_to_rejected"].append(similarity_to_rejected)
                if prompt_type:
                    prompt_types_metrics[prompt_type]["similarity_to_rejected"].append(
                        similarity_to_rejected
                    )
            else:
                similarity_to_rejected = None

            # Save the results
            results.append(
                {
                    "prompt": prompt,
                    "prompt_type": prompt_type,
                    "generated_code": generated_code,
                    "preprocess_code_blocks": preprocess_blocks,
                    "chosen_code": chosen,
                    "rejected_code": rejected,
                    "code_block_metrics": block_metrics,
                    "execution_result": execution_result,
                    "similarity_to_chosen": similarity_to_chosen,
                    "similarity_to_rejected": similarity_to_rejected,
                }
            )
    print(f"Total result length:{len(results)}")
    # Calculate aggregate metrics for overall results
    aggregated_metrics = {}
    for key in metrics:
        if key == "execution_success_rate":
            aggregated_metrics[key] = metrics[key] / len(eval_dataset)
        elif len(metrics[key]) > 0:
            aggregated_metrics[key] = np.mean(metrics[key])
        else:
            aggregated_metrics[key] = None

    # Calculate aggregate metrics for each prompt type
    aggregated_prompt_type_metrics = {}
    for p_type, p_metrics in prompt_types_metrics.items():
        aggregated_prompt_type_metrics[p_type] = {}
        for key in p_metrics:
            if key == "execution_success_rate":
                aggregated_prompt_type_metrics[p_type][key] = (
                    p_metrics[key] / prompt_types_counts[p_type]
                )
            elif len(p_metrics[key]) > 0:
                aggregated_prompt_type_metrics[p_type][key] = np.mean(p_metrics[key])
            else:
                aggregated_prompt_type_metrics[p_type][key] = None

    def convert_numpy(obj):
        """Recursively convert NumPy types to Python native types."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()  # Convert NumPy array to Python list
        elif isinstance(obj, np.generic):
            return (
                obj.item()
            )  # Convert NumPy scalars (e.g., np.float32, np.int64) to Python types
        elif isinstance(obj, dict):
            return {key: convert_numpy(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        return obj  # Return as-is if no conversion is needed

    # Convert all results and metrics
    results = convert_numpy(results)
    aggregated_metrics = convert_numpy(aggregated_metrics)
    aggregated_prompt_type_metrics = convert_numpy(aggregated_prompt_type_metrics)

    # Save results to a file
    with open(os.path.join(output_dir, "eval_results.json"), "w") as f:
        json.dump(
            {
                "results": results,
                "metrics": aggregated_metrics,
                "prompt_type_metrics": aggregated_prompt_type_metrics,
            },
            f,
            indent=2,
        )

    # Print summary of evaluation results
    print(
        f"Evaluation completed. Results saved to: {os.path.join(output_dir, 'eval_results.json')}"
    )
    print("\nOverall metrics:")
    for key, value in aggregated_metrics.items():
        print(f"  {key}: {value}")

    if aggregated_prompt_type_metrics:
        print("\nMetrics by prompt type:")
        for prompt_type, metrics in aggregated_prompt_type_metrics.items():
            print(f"\n  Prompt type: {prompt_type}")
            for key, value in metrics.items():
                print(f"    {key}: {value}")
