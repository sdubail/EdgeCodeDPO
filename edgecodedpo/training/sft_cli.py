import os
from typing import Any

import typer
import yaml
from rich.console import Console
from rich.panel import Panel
from rich.traceback import Traceback

from edgecodedpo.training.sft import train_sft

console = Console()


def load_training_config(config_path: str) -> dict[str, Any]:
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
    current = config
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def register_sft_commands(app: typer.Typer) -> None:
    """
    Register SFT training commands with the Typer app.
    """

    @app.command(
        name="sft-train", help="Train a model using Supervised Fine-Tuning (SFT)"
    )
    def train_sft_cli(
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
            help="Path to the dataset (assumed to have 'prompt' and 'chosen' columns)",
        ),
        output_dir: str = typer.Option(
            "edgecodedpo/models/sft",
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
        push_to_hub: bool | None = typer.Option(
            None,
            "--push-to-hub",
            help="Push model to HuggingFace Hub (overrides config)",
        ),
        hub_model_id: str | None = typer.Option(
            None,
            "--hub-model-id",
            help="HuggingFace Hub model ID (overrides config)",
        ),
    ) -> None:
        """
        Train a model using Supervised Fine-Tuning (SFT).
        """
        config = load_training_config(config_file)

        if model_name_or_path is None:
            model_name_or_path = get_config_value(
                config, "model", "name", default="Qwen/Qwen2-0.5B-Instruct"
            )

        use_quantization = get_config_value(
            config, "optimization", "quantization", "enabled", default=False
        )

        use_lora = get_config_value(
            config, "optimization", "lora", "enabled", default=False
        )

        num_epochs = (
            epochs
            if epochs is not None
            else get_config_value(config, "training", "num_train_epochs", default=3)
        )
        lr = (
            learning_rate
            if learning_rate is not None
            else get_config_value(config, "training", "learning_rate", default=5e-5)
        )
        bs = (
            batch_size
            if batch_size is not None
            else get_config_value(
                config, "training", "per_device_train_batch_size", default=4
            )
        )

        push_to_hub_enabled = (
            push_to_hub
            if push_to_hub is not None
            else get_config_value(config, "hub", "push_to_hub", default=False)
        )
        hub_id = (
            hub_model_id
            if hub_model_id is not None
            else get_config_value(config, "hub", "hub_model_id", default=None)
        )

        console.print(
            Panel.fit(
                "🚀 [bold blue]EdgeCodeDPO SFT Training[/bold blue]",
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
        console.print(f"  Push to Hub: [cyan]{push_to_hub_enabled}[/cyan]")
        if push_to_hub_enabled and hub_id:
            console.print(f"  Hub model ID: [cyan]{hub_id}[/cyan]")

        sft_config = {
            "num_train_epochs": num_epochs,
            "learning_rate": lr,
            "per_device_train_batch_size": bs,
            "per_device_eval_batch_size": get_config_value(
                config, "training", "per_device_eval_batch_size", default=bs
            ),
            "gradient_accumulation_steps": get_config_value(
                config, "training", "gradient_accumulation_steps", default=1
            ),
            "bf16": get_config_value(config, "training", "bf16", default=True),
            "fp16": get_config_value(config, "training", "fp16", default=False),
            "weight_decay": get_config_value(
                config, "training", "weight_decay", default=0.01
            ),
            "max_length": get_config_value(config, "model", "max_length", default=1024),
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
            "dataset_num_proc": get_config_value(
                config, "training", "dataset_num_proc", default=4
            ),
        }

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

        lora_config = None
        if use_lora:
            lora_r = get_config_value(config, "optimization", "lora", "r", default=16)
            lora_alpha = get_config_value(
                config, "optimization", "lora", "lora_alpha", default=16
            )
            lora_dropout = get_config_value(
                config, "optimization", "lora", "lora_dropout", default=0.05
            )
            lora_config = {
                "r": lora_r,
                "lora_alpha": lora_alpha,
                "lora_dropout": lora_dropout,
                "target_modules": get_config_value(
                    config, "optimization", "lora", "target_modules", default=None
                ),
            }

        try:
            train_sft(
                model_name_or_path=model_name_or_path,
                dataset_path=dataset_path,
                output_dir=output_dir,
                sft_config=sft_config,
                quantization_config=quantization_config,
                lora_config=lora_config,
                push_to_hub=push_to_hub_enabled,
                hub_model_id=hub_id,
            )

            console.print(
                Panel.fit(
                    "✅ [bold green]SFT training completed successfully![/bold green]",
                    border_style="green",
                )
            )
        except Exception as e:
            console.print(
                Panel.fit(
                    f"❌ [bold red]Error:[/bold red] {e}",
                    title="SFT Training Failed",
                    border_style="red",
                )
            )
            console.print(Traceback())
            raise typer.Exit(code=1)
