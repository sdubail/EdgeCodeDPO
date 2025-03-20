"""
Entry point for the CLI application.
"""

import asyncio
import os

import typer
from datasets import load_dataset
from rich.console import Console
from rich.panel import Panel
from rich.traceback import Traceback

from edgecodedpo.data.dataset_generator import (
    generate_dataset,
    generate_dataset_statistics,
    upload_to_huggingface,
)
from edgecodedpo.training.integration import register_training_commands

app = typer.Typer(
    name="edgecodedpo",
    help="A toolkit for generating and fine-tuning code models using Direct Preference Optimization (DPO)",
    add_completion=False,
)

console = Console()


@app.command(name="generate", help="Generate a code dataset using OpenAI API")
def generate(
    config: str = typer.Option(
        "edgecodedpo/configs/dataset.yaml",
        "--config",
        "-c",
        help="Path to the configuration file",
    ),
    output: str = typer.Option(
        "edgecodedpo/data/gen_data",
        "--output",
        "-o",
        help="Path to save the dataset",
    ),
    is_test: bool = typer.Option(
        False, "--test", "-t", help="Is the dataset for training or test purposes"
    ),
    samples: int | None = typer.Option(
        None,
        "--samples",
        "-s",
        help="Number of combinations to sample (default: all)",
    ),
    batch_size: int = typer.Option(
        5,
        "--batch-size",
        "-b",
        help="Number of concurrent API requests",
    ),
    model: str = typer.Option(
        "gpt-4o-mini",
        "--model",
        "-m",
        help="OpenAI model to use",
    ),
    system_message: str | None = typer.Option(
        None,
        "--system-message",
        help="System message for the API",
    ),
    no_intermediate: bool = typer.Option(
        False,
        "--no-intermediate",
        help="Don't save intermediate results",
    ),
) -> None:
    """
    Generate a dataset of code examples using OpenAI API.

    This command creates pairs of code examples (chosen and rejected) across various
    programming domains, tasks, and code structures for preference optimization.
    """
    console.print(
        Panel.fit(
            "🚀 [bold blue]EdgeCodeDPO Dataset Generator[/bold blue]",
            title="Starting",
            border_style="green",
        )
    )

    console.print("[bold]Configuration:[/bold]")
    if is_test:
        console.print("  Dataset is for test purposes.")
    else:
        console.print("  Dataset is for training purposes.")
    console.print(f"  Config file: [cyan]{config}[/cyan]")
    console.print(f"  Output path: [cyan]{output}[/cyan]")
    console.print(f"  Model: [cyan]{model}[/cyan]")
    console.print(f"  Batch size: [cyan]{batch_size}[/cyan]")
    console.print(f"  Samples: [cyan]{samples if samples else 'all'}[/cyan]")
    console.print(f"  Save intermediate results: [cyan]{not no_intermediate}[/cyan]")

    # Create output directory if it doesn't exist
    os.makedirs(output, exist_ok=True)

    try:
        # Run the generator
        asyncio.run(
            generate_dataset(
                config_file=config,
                output_path=output,
                is_test=is_test,
                num_samples=samples,
                batch_size=batch_size,
                openai_model=model,
                system_message=system_message,
                save_intermediate=not no_intermediate,
            )
        )

        console.print(
            Panel.fit(
                "✅ [bold green]Dataset generation completed successfully![/bold green]",
                border_style="green",
            )
        )
    except Exception as e:
        console.print(
            Panel.fit(
                f"❌ [bold red]Error:[/bold red] {e}",
                title="Dataset Generation Failed",
                border_style="red",
            )
        )
        console.print(Traceback())
        raise typer.Exit(code=1)


@app.command(name="upload", help="Upload a dataset to HuggingFace Hub")
def upload(
    dataset_path: str = typer.Option(
        "edgecodedpo/data/gen_data/dataset",
        "--dataset-path",
        "-d",
        help="Path to the saved HuggingFace dataset",
    ),
    repo_id: str = typer.Option(
        "simondubail/edgecodedpo",
        "--repo-id",
        "-r",
        help="ID for the HuggingFace repository (format: 'username/repo_name')",
    ),
    private: bool = typer.Option(
        False,
        "--private",
        "-p",
        help="Whether the repository should be private",
    ),
    token: str = typer.Option(
        None,
        "--token",
        "-t",
        help="HuggingFace API token (optional, defaults to HF_KEY in environment)",
    ),
    fuse: bool = typer.Option(
        False,
        "--fuse",
        "-f",
        help="Fuse all datasets found in gen_data* directories before uploading",
    ),
) -> None:
    """
    Upload a dataset to HuggingFace Hub.

    This command uploads a previously generated dataset to your HuggingFace account,
    making it available for sharing or fine-tuning models.
    """
    console.print(
        Panel.fit(
            "🚀 [bold blue]EdgeCodeDPO Dataset Upload[/bold blue]",
            title="Starting",
            border_style="green",
        )
    )

    console.print("[bold]Configuration:[/bold]")
    console.print(f"  Dataset path: [cyan]{dataset_path}[/cyan]")
    console.print(f"  Repository ID: [cyan]{repo_id}[/cyan]")
    console.print(f"  Private repository: [cyan]{private}[/cyan]")
    console.print(f"  Custom token provided: [cyan]{bool(token)}[/cyan]")
    console.print(f"  Fuse datasets: [cyan]{fuse}[/cyan]")

    try:
        # Run the upload function
        asyncio.run(
            upload_to_huggingface(
                dataset_path=dataset_path,
                repo_id=repo_id,
                private=private,
                hf_token=token,
                fuse_datasets=fuse,
            )
        )

        console.print(
            Panel.fit(
                f"✅ [bold green]Dataset uploaded successfully to https://huggingface.co/datasets/{repo_id}[/bold green]",
                border_style="green",
            )
        )
    except Exception as e:
        console.print(
            Panel.fit(
                f"❌ [bold red]Error:[/bold red] {e}",
                title="Dataset Upload Failed",
                border_style="red",
            )
        )
        raise typer.Exit(code=1)


@app.command(name="download", help="Download a dataset from HuggingFace Hub")
def download(
    dataset_path: str = typer.Option(
        "edgecodedpo/data/gen_data/dataset",
        "--dataset-path",
        "-d",
        help="Path where to the save the imported HuggingFace dataset",
    ),
    repo_id: str = typer.Option(
        "simondubail/edgecodedpo",
        "--repo-id",
        "-r",
        help="ID for the HuggingFace repository (format: 'username/repo_name')",
    ),
    token: str = typer.Option(
        None,
        "--token",
        "-t",
        help="HuggingFace API token (optional, defaults to HF_KEY in environment)",
    ),
) -> None:
    """
    Download a dataset from HuggingFace Hub.

    This command download a previously generated and uploaded dataset to your local machine.
    """
    console.print(
        Panel.fit(
            "🚀 [bold blue]EdgeCodeDPO Dataset Download[/bold blue]",
            title="Starting",
            border_style="green",
        )
    )

    console.print("[bold]Configuration:[/bold]")
    console.print(f"  Dataset path: [cyan]{dataset_path}[/cyan]")
    console.print(f"  Repository ID: [cyan]{repo_id}[/cyan]")
    console.print(f"  Custom token provided: [cyan]{bool(token)}[/cyan]")

    try:
        dataset = load_dataset(repo_id)
        os.makedirs(dataset_path, exist_ok=True)
        dataset.save_to_disk(dataset_path)

        console.print(
            Panel.fit(
                f"✅ [bold green]Dataset downloaded successfully to {dataset_path}[/bold green]",
                border_style="green",
            )
        )
    except Exception as e:
        console.print(
            Panel.fit(
                f"❌ [bold red]Error:[/bold red] {e}",
                title="Dataset Download Failed",
                border_style="red",
            )
        )
        raise typer.Exit(code=1)


@app.command(name="stats", help="Generate token length statistics for a dataset")
def stats(
    dataset_path: str = typer.Option(
        "simondubail/edgecodedpo",
        "--dataset",
        "-d",
        help="Path to the dataset or HuggingFace dataset ID",
    ),
    tokenizer: str = typer.Option(
        "Qwen/Qwen2-0.5B-Instruct",
        "--tokenizer",
        "-t",
        help="Tokenizer to use for tokenization",
    ),
    output: str = typer.Option(
        "edgecodedpo/data/stats",
        "--output",
        "-o",
        help="Path to save the statistics and figures",
    ),
    cpu_only: bool = typer.Option(
        False,
        "--cpu-only",
        help="Use CPU only for processing",
    ),
    batch_size: int = typer.Option(
        32,
        "--batch-size",
        "-b",
        help="Batch size for processing",
    ),
) -> None:
    """
    Generate token length statistics for a dataset.

    This command analyzes the distribution of token lengths for prompts, chosen, and rejected
    completions in a dataset and generates visualizations.
    """
    console.print(
        Panel.fit(
            "📊 [bold blue]EdgeCodeDPO Dataset Statistics Generator[/bold blue]",
            title="Starting",
            border_style="green",
        )
    )

    console.print("[bold]Configuration:[/bold]")
    console.print(f"  Dataset path: [cyan]{dataset_path}[/cyan]")
    console.print(f"  Tokenizer: [cyan]{tokenizer}[/cyan]")
    console.print(f"  Output path: [cyan]{output}[/cyan]")
    console.print(f"  CPU only: [cyan]{cpu_only}[/cyan]")
    console.print(f"  Batch size: [cyan]{batch_size}[/cyan]")

    # Create output directory if it doesn't exist
    os.makedirs(output, exist_ok=True)

    try:
        # Run the statistics generator
        stats_result = asyncio.run(
            generate_dataset_statistics(
                dataset_path=dataset_path,
                tokenizer_name_or_path=tokenizer,
                output_dir=output,
                use_gpu=not cpu_only,
                batch_size=batch_size,
            )
        )

        # Display some key statistics
        console.print("\n[bold]Key Statistics:[/bold]")
        console.print("  [bold]Prompt token lengths:[/bold]")
        console.print(f"    Mean: [cyan]{stats_result['prompt']['mean']:.1f}[/cyan]")
        console.print(
            f"    Median: [cyan]{stats_result['prompt']['median']:.1f}[/cyan]"
        )
        console.print(
            f"    95th percentile: [cyan]{stats_result['prompt']['q95']:.1f}[/cyan]"
        )
        console.print(f"    Max: [cyan]{stats_result['prompt']['max']:.1f}[/cyan]")

        console.print("  [bold]Chosen completion token lengths:[/bold]")
        console.print(f"    Mean: [cyan]{stats_result['chosen']['mean']:.1f}[/cyan]")
        console.print(
            f"    Median: [cyan]{stats_result['chosen']['median']:.1f}[/cyan]"
        )
        console.print(
            f"    95th percentile: [cyan]{stats_result['chosen']['q95']:.1f}[/cyan]"
        )
        console.print(f"    Max: [cyan]{stats_result['chosen']['max']:.1f}[/cyan]")

        console.print("  [bold]Rejected completion token lengths:[/bold]")
        console.print(f"    Mean: [cyan]{stats_result['rejected']['mean']:.1f}[/cyan]")
        console.print(
            f"    Median: [cyan]{stats_result['rejected']['median']:.1f}[/cyan]"
        )
        console.print(
            f"    95th percentile: [cyan]{stats_result['rejected']['q95']:.1f}[/cyan]"
        )
        console.print(f"    Max: [cyan]{stats_result['rejected']['max']:.1f}[/cyan]")

        console.print(
            Panel.fit(
                f"✅ [bold green]Statistics generation completed successfully![/bold green]\n"
                f"Token length distributions saved to {output}/token_length_distributions.png\n"
                f"Statistics saved to {output}/token_length_stats.json",
                border_style="green",
            )
        )
    except Exception as e:
        console.print(
            Panel.fit(
                f"❌ [bold red]Error:[/bold red] {e}",
                title="Statistics Generation Failed",
                border_style="red",
            )
        )
        console.print(Traceback())
        raise typer.Exit(code=1)


@app.command(name="version")
def version() -> None:
    """Display the current version of EdgeCodeDPO."""
    console.print(
        "[bold]EdgeCodeDPO[/bold] version: [cyan]0.1.0[/cyan] (early development)"
    )


# Register the training commands
register_training_commands(app)


def main() -> None:
    """Entry point for the CLI application."""
    app()


if __name__ == "__main__":
    main()
