import os
from typing import Any

from datasets import load_dataset, load_from_disk
from transformers import (
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from trl import DPOConfig

from edgecodedpo.config import settings
from edgecodedpo.training.dpo import load_model_and_tokenizer
from edgecodedpo.training.visualization import create_visualizations


def preprocess_sft_dataset(
    dataset_path: str,
    tokenizer,
    max_length: int = 1024,
    num_proc: int = 4,
    combine_prompt_and_chosen: bool = True,
):
    """
    Loads a dataset that has 'prompt', 'chosen', (and maybe 'rejected')
    but uses only the 'chosen' data for supervised fine-tuning.
    It will return a dataset of tokenized samples for standard Causal LM.

    Args:
        dataset_path: Local path or HF Hub path to dataset.
        tokenizer: The tokenizer to process text into tokens.
        max_length: Maximum token length for each sample.
        num_proc: Number of processes for .map() if you use it.
        combine_prompt_and_chosen: Whether to concatenate the prompt + chosen
                                   as the full training text.

    Returns:
        A tokenized Dataset suitable for SFT.
    """
    # ---------------------------------------------------------------------
    # 1A. Load the dataset from disk or from the HF Hub
    # ---------------------------------------------------------------------
    if os.path.exists(dataset_path):
        dataset = load_from_disk(dataset_path)
    else:
        # Attempt to load from HuggingFace Hub
        if ":" in dataset_path:
            repo_id, split = dataset_path.split(":", 1)
            dataset = load_dataset(repo_id, split=split)
        else:
            dataset_obj = load_dataset(dataset_path)
            # If the dataset is a DatasetDict with multiple splits, prefer 'train'
            if hasattr(dataset_obj, "keys") and "train" in dataset_obj:
                dataset = dataset_obj["train"]
            else:
                dataset = dataset_obj

    # ---------------------------------------------------------------------
    # 1B. Create the "text" to be used for training from 'prompt' + 'chosen'
    # ---------------------------------------------------------------------
    # In SFT, train the model to produce the chosen answer,
    # potentially conditioned on the prompt. If your dataset has columns:
    #   - 'prompt'
    #   - 'chosen' (the better response)

    def build_text(example):
        prompt = example.get("prompt", "")[0]["content"]
        chosen = example.get("chosen", "")[0]["content"]
        if isinstance(chosen, list):
            chosen = chosen[0] if chosen else ""
        if combine_prompt_and_chosen:
            return prompt + "\n" + chosen
        else:
            return chosen

    # ---------------------------------------------------------------------
    # 1C. Tokenize each sample
    # ---------------------------------------------------------------------
    def tokenize_fn(example):
        text = build_text(example)
        tokens = tokenizer(
            text,
            max_length=max_length,
            truncation=True,
        )
        return tokens

    dataset = dataset.map(tokenize_fn, batched=False, num_proc=num_proc)

    return dataset


##############################################################################
# 2) The main SFT training function
##############################################################################
def train_sft(
    model_name_or_path: str,
    dataset_path: str,
    output_dir: str,
    sft_config: dict[str, Any] | None = None,
    quantization_config: dict[str, Any] | None = None,
    lora_config: dict[str, Any] | None = None,
    push_to_hub: bool = False,
    hub_model_id: str | None = None,
):
    """
    Train a language model via standard supervised fine-tuning (SFT).
    This is similar in style to 'train_dpo', but it uses
    cross-entropy on the 'chosen' data instead of DPO.

    Args:
        model_name_or_path: Hugging Face model name or local path
        dataset_path: Path or HF Hub ID to the dataset
        output_dir: Directory to save the trained model
        sft_config: Dict of SFT training hyperparams (epochs, LR, etc.)
        quantization_config: 4-bit/8-bit loading config
        lora_config: LoRA config
        push_to_hub: Whether to push model to HF Hub
        hub_model_id: The Hub repo name if push_to_hub is True
    """
    # ---------------------------------------------------------------------
    # 2A. Create output dir
    # ---------------------------------------------------------------------
    os.makedirs(output_dir, exist_ok=True)

    # ---------------------------------------------------------------------
    # 2B. Set default SFT config if none provided
    # ---------------------------------------------------------------------
    if sft_config is None:
        sft_config = {}
    num_train_epochs = int(sft_config.get("num_train_epochs", 3))
    train_batch_size = int(sft_config.get("per_device_train_batch_size", 2))
    eval_batch_size = int(sft_config.get("per_device_eval_batch_size", 2))
    learning_rate = float(sft_config.get("learning_rate", 1e-5))
    warmup_ratio = float(sft_config.get("warmup_ratio", 0.03))
    logging_steps = int(sft_config.get("logging_steps", 50))
    save_steps = int(sft_config.get("save_steps", 500))
    eval_steps = int(sft_config.get("eval_steps", 500))
    max_length = int(sft_config.get("max_length", 1024))
    bf16 = bool(sft_config.get("bf16", False))
    fp16 = bool(sft_config.get("fp16", True))
    eval_split = float(sft_config.get("eval_split", 0.1))
    dataset_num_proc = int(sft_config.get("dataset_num_proc", 4))

    # ---------------------------------------------------------------------
    # 2C. Load model and tokenizer
    # ---------------------------------------------------------------------

    model, tokenizer = load_model_and_tokenizer(
        model_name_or_path=model_name_or_path,
        dpo_config=DPOConfig(
            output_dir=output_dir
        ),  # We won't really use it for SFT, but we keep the signature
        quantization_config=quantization_config,
        lora_config=lora_config,
    )

    # ---------------------------------------------------------------------
    # 2D. Load & preprocess dataset (only using 'chosen' for SFT)
    # ---------------------------------------------------------------------
    dataset = preprocess_sft_dataset(
        dataset_path=dataset_path,
        tokenizer=tokenizer,
        max_length=max_length,
        num_proc=dataset_num_proc,
    )

    # ---------------------------------------------------------------------
    # 2E. Create train & eval splits if needed
    # ---------------------------------------------------------------------
    if "train" not in dataset and "test" not in dataset:
        dataset = dataset.train_test_split(test_size=eval_split)
        train_dataset = dataset["train"]
        eval_dataset = dataset["test"]
    else:
        # if there's a built-in train/test
        train_dataset = dataset.get("train", dataset)
        eval_dataset = dataset.get("test", None)

    # ---------------------------------------------------------------------
    # 2F. Set up HF TrainingArguments for SFT
    # ---------------------------------------------------------------------
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        evaluation_strategy="steps" if eval_dataset else "no",
        eval_steps=eval_steps,
        save_steps=save_steps,
        logging_steps=logging_steps,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        bf16=bf16,
        fp16=fp16,
        push_to_hub=push_to_hub,
        hub_model_id=hub_model_id,
        hub_token=settings.HF_KEY if push_to_hub else None,
    )

    # ---------------------------------------------------------------------
    # 2G. Create a data collator for causal language modeling
    # ---------------------------------------------------------------------
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # We don't want masked language modeling
    )

    # ---------------------------------------------------------------------
    # 2H. Create a standard Trainer for SFT
    # ---------------------------------------------------------------------
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    # ---------------------------------------------------------------------
    # 2I. Train!
    # ---------------------------------------------------------------------
    train_result = trainer.train()

    # ---------------------------------------------------------------------
    # 2J. Save the final model & metrics
    # ---------------------------------------------------------------------
    trainer.save_model(output_dir)
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    # ---------------------------------------------------------------------
    # 2K. Generate any visualizations if you want to reuse your existing code
    # ---------------------------------------------------------------------
    log_dir = os.path.join(output_dir, "runs")  # HF Trainer logs go in /runs by default
    viz_dir = os.path.join(output_dir, "visualizations")

    try:
        print("Generating metric visualizations...")
        create_visualizations(log_dir=log_dir, output_dir=viz_dir)
        print(f"Visualizations saved to {viz_dir}")
    except Exception as e:
        print(f"Warning: Could not generate visualizations: {e}")

    # ---------------------------------------------------------------------
    # 2L. Optionally push to hub
    # ---------------------------------------------------------------------
    if push_to_hub and hub_model_id:
        trainer.push_to_hub()
        print(
            f"Model pushed to Hugging Face Hub at: https://huggingface.co/{hub_model_id}"
        )

    print(f"SFT complete! Model saved to: {output_dir}")
