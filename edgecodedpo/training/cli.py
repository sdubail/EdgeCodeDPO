"""
Add DPO training commands to the EdgeCodeDPO CLI with YAML configuration support.
"""

import os
from typing import Any

import typer
import yaml
from rich.console import Console
from rich.panel import Panel
from rich.traceback import Traceback

from edgecodedpo.training.dpo import load_and_evaluate_model, train_dpo

console = Console()


def load_training_config(config_path: str) -> dict[str, Any]:
    """
    Load the training configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file

    Returns:
        Dictionary containing the configuration
    """
    if not os.path.exists(config_path):
        console.print(
            f"[bold red]Warning:[/bold red] Config file {config_path} not found. Using default values."
        )
        return {}

    with open(config_path) as file:
        try:
            config = yaml.safe_load(file)
            return config
        except yaml.YAMLError as e:
            console.print(f"[bold red]Error parsing YAML configuration:[/bold red] {e}")
            return {}


def get_config_value(config: dict[str, Any], *keys: str, default: Any = None) -> Any:
    """
    Get a value from a nested dictionary using a list of keys.

    Args:
        config: The configuration dictionary
        keys: Sequence of keys to navigate the nested dictionary
        default: Default value to return if the key is not found

    Returns:
        The configuration value or default if not found
    """
    current = config
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def register_dpo_commands(app: typer.Typer) -> None:
    """
    Register DPO training commands with the Typer app.

    Args:
        app: The Typer app to register commands with
    """

    @app.command(
        name="train", help="Train a model using Direct Preference Optimization (DPO)"
    )
    def train(
        model_name_or_path: str = typer.Option(
            None,
            "--model",
            "-m",
            help="Base model or fine-tuned model to use (overrides config)",
        ),
        dataset_path: str = typer.Option(
            "edgecodedpo/data/gen_data/dataset",
            "--dataset",
            "-d",
            help="Path to the dataset (must have 'chosen' and 'rejected' columns)",
        ),
        output_dir: str = typer.Option(
            "edgecodedpo/models/dpo",
            "--output",
            "-o",
            help="Directory to save the model",
        ),
        config_file: str = typer.Option(
            "edgecodedpo/configs/training.yaml",
            "--config",
            "-c",
            help="Path to the training configuration file",
        ),
        # Expose only a few critical parameters as CLI overrides
        epochs: int | None = typer.Option(
            None,
            "--epochs",
            "-e",
            help="Number of training epochs (overrides config)",
        ),
        learning_rate: float | None = typer.Option(
            None,
            "--learning-rate",
            "-lr",
            help="Learning rate (overrides config)",
        ),
        batch_size: int | None = typer.Option(
            None,
            "--batch-size",
            "-b",
            help="Per-device training batch size (overrides config)",
        ),
        beta: float | None = typer.Option(
            None,
            "--beta",
            help="DPO beta parameter (overrides config)",
        ),
        loss_type: str | None = typer.Option(
            None,
            "--loss-type",
            help="DPO loss type (overrides config)",
        ),
        push_to_hub: bool | None = typer.Option(
            None,
            "--push-to-hub",
            help="Push model to HuggingFace Hub (overrides config)",
        ),
        is_already_lora: bool = typer.Option(
            False,
            "--is_already_lora",
            help="Push model to HuggingFace Hub (overrides config)",
        ),
        hub_model_id: str | None = typer.Option(
            None,
            "--hub-model-id",
            help="HuggingFace Hub model ID (overrides config)",
        ),
    ) -> None:
        """
        Train a model using Direct Preference Optimization (DPO).
        """
        # Load the training configuration
        config = load_training_config(config_file)

        # Get model name from config if not provided
        if model_name_or_path is None:
            model_name_or_path = get_config_value(
                config, "model", "name", default="Qwen/Qwen2-0.5B-Instruct"
            )

        # Determine quantization settings from config
        use_quantization = get_config_value(
            config, "optimization", "quantization", "enabled", default=False
        )

        # Determine LoRA settings from config
        use_lora = get_config_value(
            config, "optimization", "lora", "enabled", default=False
        )

        # Get or override epochs from config
        num_epochs = (
            epochs
            if epochs is not None
            else get_config_value(config, "training", "num_train_epochs", default=3)
        )

        # Get or override learning rate from config
        lr = (
            learning_rate
            if learning_rate is not None
            else get_config_value(config, "training", "learning_rate", default=5e-5)
        )

        # Get or override batch size from config
        bs = (
            batch_size
            if batch_size is not None
            else get_config_value(
                config, "training", "per_device_train_batch_size", default=4
            )
        )

        # Get or override beta from config
        dpo_beta = (
            beta
            if beta is not None
            else get_config_value(config, "dpo", "beta", default=0.1)
        )

        # Get or override loss type from config
        dpo_loss_type = (
            loss_type
            if loss_type is not None
            else get_config_value(config, "dpo", "loss_type", default="sigmoid")
        )

        # Get or override push to hub from config
        push_to_hub_enabled = (
            push_to_hub
            if push_to_hub is not None
            else get_config_value(config, "hub", "push_to_hub", default=False)
        )

        # Get or override hub model ID from config
        hub_id = (
            hub_model_id
            if hub_model_id is not None
            else get_config_value(config, "hub", "hub_model_id", default=None)
        )

        console.print(
            Panel.fit(
                "🚀 [bold blue]EdgeCodeDPO DPO Training[/bold blue]",
                title="Starting",
                border_style="green",
            )
        )

        console.print("[bold]Configuration:[/bold]")
        console.print(f"  Config file: [cyan]{config_file}[/cyan]")
        console.print(f"  Base model: [cyan]{model_name_or_path}[/cyan]")
        console.print(f"  Dataset path: [cyan]{dataset_path}[/cyan]")
        console.print(f"  Output directory: [cyan]{output_dir}[/cyan]")
        console.print(f"  Epochs: [cyan]{num_epochs}[/cyan]")
        console.print(f"  Learning rate: [cyan]{lr}[/cyan]")
        console.print(f"  Batch size: [cyan]{bs}[/cyan]")
        console.print(f"  Using Quantization: [cyan]{use_quantization}[/cyan]")
        console.print(f"  Using LoRA: [cyan]{use_lora}[/cyan]")

        if use_lora:
            lora_r = get_config_value(config, "optimization", "lora", "r", default=16)
            lora_alpha = get_config_value(
                config, "optimization", "lora", "lora_alpha", default=16
            )
            lora_dropout = get_config_value(
                config, "optimization", "lora", "lora_dropout", default=0.05
            )
            console.print(f"  LoRA rank (r): [cyan]{lora_r}[/cyan]")
            console.print(f"  LoRA alpha: [cyan]{lora_alpha}[/cyan]")
            console.print(f"  LoRA dropout: [cyan]{lora_dropout}[/cyan]")

        console.print(f"  DPO beta: [cyan]{dpo_beta}[/cyan]")
        console.print(f"  DPO loss type: [cyan]{dpo_loss_type}[/cyan]")
        console.print(f"  Push to Hub: [cyan]{push_to_hub_enabled}[/cyan]")
        if push_to_hub_enabled and hub_id:
            console.print(f"  Hub model ID: [cyan]{hub_id}[/cyan]")

        # Prepare DPO config from the YAML config
        dpo_config = {
            "num_train_epochs": num_epochs,
            "learning_rate": lr,
            "per_device_train_batch_size": bs,
            "per_device_eval_batch_size": get_config_value(
                config, "training", "per_device_eval_batch_size", default=bs
            ),
            "gradient_accumulation_steps": get_config_value(
                config, "training", "gradient_accumulation_steps", default=1
            ),
            "beta": dpo_beta,
            "loss_type": dpo_loss_type,
            "bf16": get_config_value(config, "training", "bf16", default=True),
            "fp16": get_config_value(config, "training", "fp16", default=False),
            "weight_decay": get_config_value(
                config, "training", "weight_decay", default=0.01
            ),
            "max_length": get_config_value(config, "model", "max_length", default=1024),
            "max_prompt_length": get_config_value(
                config, "model", "max_prompt_length", default=512
            ),
            "logging_steps": get_config_value(
                config, "training", "logging_steps", default=10
            ),
            "save_steps": get_config_value(
                config, "training", "save_steps", default=100
            ),
            "eval_steps": get_config_value(
                config, "training", "eval_steps", default=100
            ),
            "eval_split": get_config_value(
                config, "training", "eval_split", default=0.1
            ),
            "save_total_limit": get_config_value(
                config, "training", "save_total_limit", default=3
            ),
            "dataset_num_proc": get_config_value(
                config, "training", "dataset_num_proc", default=4
            ),
            # Advanced DPO settings
            "reference_free": get_config_value(
                config, "dpo", "reference_free", default=False
            ),
            "label_smoothing": get_config_value(
                config, "dpo", "label_smoothing", default=0.0
            ),
            "generate_during_eval": get_config_value(
                config, "dpo", "generate_during_eval", default=True
            ),
            # Advanced sync settings
            "sync_ref_model": get_config_value(
                config, "advanced_dpo", "sync_ref_model", default=False
            ),
            "ref_model_mixup_alpha": get_config_value(
                config, "advanced_dpo", "ref_model_mixup_alpha", default=0.6
            ),
            "ref_model_sync_steps": get_config_value(
                config, "advanced_dpo", "ref_model_sync_steps", default=512
            ),
            "precompute_ref_log_probs": get_config_value(
                config, "advanced_dpo", "precompute_ref_log_probs", default=False
            ),
            "rpo_alpha": get_config_value(
                config, "advanced_dpo", "rpo_alpha", default=None
            ),
            "use_weighting": get_config_value(
                config, "advanced_dpo", "use_weighting", default=False
            ),
        }

        # Quantization config from the YAML config
        quantization_config = None
        if use_quantization:
            quantization_config = {
                "load_in_4bit": get_config_value(
                    config, "optimization", "quantization", "load_in_4bit", default=True
                ),
                "bnb_4bit_use_double_quant": get_config_value(
                    config,
                    "optimization",
                    "quantization",
                    "bnb_4bit_use_double_quant",
                    default=True,
                ),
                "bnb_4bit_quant_type": get_config_value(
                    config,
                    "optimization",
                    "quantization",
                    "bnb_4bit_quant_type",
                    default="nf4",
                ),
                "bnb_4bit_compute_dtype": get_config_value(
                    config,
                    "optimization",
                    "quantization",
                    "bnb_4bit_compute_dtype",
                    default="bfloat16",
                ),
            }

        # LoRA config from the YAML config
        lora_config = None
        if use_lora:
            lora_config = {
                "r": get_config_value(config, "optimization", "lora", "r", default=16),
                "lora_alpha": get_config_value(
                    config, "optimization", "lora", "lora_alpha", default=16
                ),
                "lora_dropout": get_config_value(
                    config, "optimization", "lora", "lora_dropout", default=0.05
                ),
                "target_modules": get_config_value(
                    config, "optimization", "lora", "target_modules", default=None
                ),
            }

        try:
            # Train model
            train_dpo(
                model_name_or_path=model_name_or_path,
                dataset_path=dataset_path,
                output_dir=output_dir,
                dpo_config=dpo_config,
                quantization_config=quantization_config,
                lora_config=lora_config,
                is_already_lora=is_already_lora,
                push_to_hub=push_to_hub_enabled,
                hub_model_id=hub_id,
            )

            console.print(
                Panel.fit(
                    "✅ [bold green]DPO training completed successfully![/bold green]",
                    border_style="green",
                )
            )
        except Exception as e:
            console.print(
                Panel.fit(
                    f"❌ [bold red]Error:[/bold red] {e}",
                    title="DPO Training Failed",
                    border_style="red",
                )
            )
            console.print(Traceback())
            raise typer.Exit(code=1)

    @app.command(name="evaluate", help="Evaluate a DPO-trained model")
    def evaluate(
        model_path: str = typer.Option(
            ...,
            "--model",
            "-m",
            help="Path to the trained model",
        ),
        dataset_path: str = typer.Option(
            ...,
            "--dataset",
            "-d",
            help="Path to the evaluation dataset",
        ),
        is_test: bool = typer.Option(
            False, "--test", "-t", help="Is the dataset for training or test purposes"
        ),
        output_dir: str = typer.Option(
            "edgecodedpo/models/evaluation",
            "--output",
            "-o",
            help="Directory to save evaluation results",
        ),
        num_examples: int = typer.Option(
            10,
            "--num-examples",
            "-n",
            help="Number of examples to evaluate",
        ),
        batch_size: int = typer.Option(
            4,
            "--batch_size",
            "-b",
            help="Batch size for parallel generation",
        ),
    ) -> None:
        """
        Evaluate a DPO-trained model.
        """
        console.print(
            Panel.fit(
                "🔍 [bold blue]EdgeCodeDPO Model Evaluation[/bold blue]",
                title="Starting",
                border_style="green",
            )
        )

        console.print("[bold]Configuration:[/bold]")
        console.print(f"  Model path: [cyan]{model_path}[/cyan]")
        console.print(f"  Dataset path: [cyan]{dataset_path}[/cyan]")
        console.print(f"  Output directory: [cyan]{output_dir}[/cyan]")
        console.print(f"  Number of examples: [cyan]{num_examples}[/cyan]")

        try:
            # Evaluate model
            load_and_evaluate_model(
                model_path=model_path,
                dataset_path=dataset_path,
                is_test=is_test,
                output_dir=output_dir,
                num_examples=num_examples,
                batch_size=batch_size,
            )

            console.print(
                Panel.fit(
                    "✅ [bold green]Model evaluation completed successfully![/bold green]",
                    border_style="green",
                )
            )
        except Exception as e:
            console.print(
                Panel.fit(
                    f"❌ [bold red]Error:[/bold red] {e}",
                    title="Model Evaluation Failed",
                    border_style="red",
                )
            )
            console.print(Traceback())
            raise typer.Exit(code=1)
