import asyncio
import os

import typer
from rich.console import Console
from rich.panel import Panel

from edgecodedpo.data.dataset_generator import generate_dataset

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
        raise typer.Exit(code=1)


@app.command(name="version")
def version() -> None:
    """Display the current version of EdgeCodeDPO."""
    console.print(
        "[bold]EdgeCodeDPO[/bold] version: [cyan]0.1.0[/cyan] (early development)"
    )


def main() -> None:
    """Entry point for the CLI application."""
    app()


if __name__ == "__main__":
    main()
