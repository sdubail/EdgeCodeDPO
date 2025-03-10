"""
DPO training module for EdgeCodeDPO.

This module provides functionality to train language models using Direct Preference Optimization (DPO)
as described in the paper "Direct Preference Optimization: Your Language Model is Secretly a Reward Model"
by Rafailov et al.
"""

import os
from typing import Any

import datasets
import torch
import typer
from datasets import Dataset, load_dataset, load_from_disk
from huggingface_hub import HfApi, login
from peft import AutoPeftModelForCausalLM, LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import DPOConfig, DPOTrainer

from edgecodedpo.config import settings


def load_model_and_tokenizer(
    model_name_or_path: str,
    dpo_config: DPOConfig,
    quantization_config: dict[str, Any] | None = None,
    lora_config: dict[str, Any] | None = None,
) -> tuple:
    """
    Load the model and tokenizer for DPO training.

    Args:
        model_name_or_path: Name or path of the base model or fine-tuned model
        dpo_config: DPO configuration
        quantization_config: Configuration for quantization (BitsAndBytes)
        lora_config: Configuration for LoRA

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
        push_to_hub: Whether to push the model to HuggingFace Hub
        hub_model_id: ID for the HuggingFace repository
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Set up DPO configuration
    training_args = DPOConfig(
        output_dir=output_dir,
        num_train_epochs=dpo_config.get("num_train_epochs", 3),
        per_device_train_batch_size=dpo_config.get("per_device_train_batch_size", 4),
        per_device_eval_batch_size=dpo_config.get("per_device_eval_batch_size", 4),
        gradient_accumulation_steps=dpo_config.get("gradient_accumulation_steps", 1),
        learning_rate=dpo_config.get("learning_rate", 5e-5),
        lr_scheduler_type=dpo_config.get("lr_scheduler_type", "cosine"),
        warmup_ratio=dpo_config.get("warmup_ratio", 0.1),
        logging_steps=dpo_config.get("logging_steps", 10),
        save_strategy=dpo_config.get("save_strategy", "steps"),
        save_steps=dpo_config.get("save_steps", 100),
        evaluation_strategy=dpo_config.get("evaluation_strategy", "steps"),
        eval_steps=dpo_config.get("eval_steps", 100),
        beta=dpo_config.get("beta", 0.1),
        loss_type=dpo_config.get("loss_type", "sigmoid"),
        max_length=dpo_config.get("max_length", 1024),
        max_prompt_length=dpo_config.get("max_prompt_length", 512),
        bf16=dpo_config.get("bf16", True),
        fp16=dpo_config.get("fp16", False),
        logging_dir=os.path.join(output_dir, "logs"),
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
    max_length: int = 1024,
    num_examples: int = 10,
) -> None:
    """
    Load a trained model and evaluate it on the dataset.

    Args:
        model_path: Path to the trained model
        dataset_path: Path to the dataset
        output_dir: Directory to save the evaluation results
        max_length: Maximum sequence length
        num_examples: Number of examples to evaluate
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
    dataset = load_from_disk(dataset_path)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Select evaluation examples
    if "test" in dataset:
        eval_dataset = dataset["test"]
    else:
        eval_dataset = dataset.select(range(min(num_examples, len(dataset))))

    # Generate predictions for the selected examples
    results = []

    for i, example in enumerate(
        eval_dataset.select(range(min(num_examples, len(eval_dataset))))
    ):
        if i >= num_examples:
            break

        # Get prompt from the example
        if "prompt" in example:
            prompt = example["prompt"]
        else:
            # Try to extract prompt from the conversation format
            chosen_convo = example.get("chosen", [])
            if (
                chosen_convo
                and isinstance(chosen_convo, list)
                and len(chosen_convo) > 0
            ):
                prompt = chosen_convo[0].get("content", "")
            else:
                prompt = "No prompt found in example"

        # Generate completion using the model
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
        output = model.generate(
            input_ids=input_ids,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.7,
        )

        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

        # Save the results
        results.append(
            {
                "prompt": prompt,
                "generated": generated_text,
                "chosen": example.get("chosen", ""),
                "rejected": example.get("rejected", ""),
            }
        )

    # Save results to a file
    import json

    with open(os.path.join(output_dir, "eval_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(
        f"Evaluation completed. Results saved to: {os.path.join(output_dir, 'eval_results.json')}"
    )
