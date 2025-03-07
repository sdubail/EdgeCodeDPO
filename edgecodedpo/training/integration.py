"""
Integration of DPO training commands into the main CLI.
"""

import typer

from edgecodedpo.training.cli import register_dpo_commands

# Create a sub-app for training
train_app = typer.Typer(
    name="train",
    help="Train and fine-tune models using various techniques",
    add_completion=False,
)

# Register DPO commands
register_dpo_commands(train_app)


def register_training_commands(app: typer.Typer) -> None:
    """
    Register the training app with the main app.

    Args:
        app: The main Typer app
    """
    app.add_typer(train_app)
