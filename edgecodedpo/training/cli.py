"""
Add DPO training commands to the EdgeCodeDPO CLI.
"""

import typer
from rich.console import Console
from rich.panel import Panel

from edgecodedpo.training.dpo import load_and_evaluate_model, train_dpo

console = Console()


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
            ...,
            "--model",
            "-m",
            help="Base model or fine-tuned model to use (e.g., 'Qwen/Qwen2-0.5B-Instruct')",
        ),
        dataset_path: str = typer.Option(
            ...,
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
        epochs: int = typer.Option(
            3,
            "--epochs",
            "-e",
            help="Number of training epochs",
        ),
        learning_rate: float = typer.Option(
            5e-5,
            "--learning-rate",
            "-lr",
            help="Learning rate",
        ),
        batch_size: int = typer.Option(
            4,
            "--batch-size",
            "-b",
            help="Per-device training batch size",
        ),
        use_qlora: bool = typer.Option(
            False,
            "--qlora",
            help="Use QLoRA for training (4-bit quantization with LoRA)",
        ),
        use_lora: bool = typer.Option(
            False,
            "--lora",
            help="Use LoRA for training",
        ),
        lora_r: int = typer.Option(
            16,
            "--lora-r",
            help="LoRA attention dimension",
        ),
        lora_alpha: int = typer.Option(
            16,
            "--lora-alpha",
            help="LoRA alpha parameter",
        ),
        lora_dropout: float = typer.Option(
            0.05,
            "--lora-dropout",
            help="LoRA dropout",
        ),
        beta: float = typer.Option(
            0.1,
            "--beta",
            help="DPO beta parameter that controls deviation from reference model",
        ),
        loss_type: str = typer.Option(
            "sigmoid",
            "--loss-type",
            help="DPO loss type (sigmoid, hinge, ipo, etc.)",
        ),
        push_to_hub: bool = typer.Option(
            False,
            "--push-to-hub",
            help="Push model to HuggingFace Hub",
        ),
        hub_model_id: str = typer.Option(
            None,
            "--hub-model-id",
            help="HuggingFace Hub model ID (format: 'username/model_name')",
        ),
    ) -> None:
        """
        Train a model using Direct Preference Optimization (DPO).
        """
        console.print(
            Panel.fit(
                "🚀 [bold blue]EdgeCodeDPO DPO Training[/bold blue]",
                title="Starting",
                border_style="green",
            )
        )

        console.print("[bold]Configuration:[/bold]")
        console.print(f"  Base model: [cyan]{model_name_or_path}[/cyan]")
        console.print(f"  Dataset path: [cyan]{dataset_path}[/cyan]")
        console.print(f"  Output directory: [cyan]{output_dir}[/cyan]")
        console.print(f"  Epochs: [cyan]{epochs}[/cyan]")
        console.print(f"  Learning rate: [cyan]{learning_rate}[/cyan]")
        console.print(f"  Batch size: [cyan]{batch_size}[/cyan]")
        console.print(f"  Using QLoRA: [cyan]{use_qlora}[/cyan]")
        console.print(f"  Using LoRA: [cyan]{use_lora}[/cyan]")
        if use_lora or use_qlora:
            console.print(f"  LoRA rank (r): [cyan]{lora_r}[/cyan]")
            console.print(f"  LoRA alpha: [cyan]{lora_alpha}[/cyan]")
            console.print(f"  LoRA dropout: [cyan]{lora_dropout}[/cyan]")
        console.print(f"  DPO beta: [cyan]{beta}[/cyan]")
        console.print(f"  DPO loss type: [cyan]{loss_type}[/cyan]")
        console.print(f"  Push to Hub: [cyan]{push_to_hub}[/cyan]")
        if push_to_hub:
            console.print(f"  Hub model ID: [cyan]{hub_model_id}[/cyan]")

        # Prepare configs
        dpo_config = {
            "num_train_epochs": epochs,
            "learning_rate": learning_rate,
            "per_device_train_batch_size": batch_size,
            "per_device_eval_batch_size": batch_size,
            "beta": beta,
            "loss_type": loss_type,
            "bf16": True,  # Use bfloat16 for training
            "logging_steps": 10,
            "save_steps": 100,
            "eval_steps": 100,
        }

        # Quantization config
        quantization_config = None
        if use_qlora:
            quantization_config = {
                "load_in_4bit": True,
                "bnb_4bit_use_double_quant": True,
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_compute_dtype": "bfloat16",
            }

        # LoRA config
        lora_config = None
        if use_lora or use_qlora:
            lora_config = {
                "r": lora_r,
                "lora_alpha": lora_alpha,
                "lora_dropout": lora_dropout,
                # You may need to adjust target modules based on the model architecture
                "target_modules": None,  # Will be automatically detected
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
                push_to_hub=push_to_hub,
                hub_model_id=hub_model_id,
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
                output_dir=output_dir,
                num_examples=num_examples,
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
            raise typer.Exit(code=1)
