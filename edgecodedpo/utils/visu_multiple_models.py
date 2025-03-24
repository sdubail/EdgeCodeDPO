"""
Visualization module for extracting and plotting TensorBoard metrics from multiple model runs.
Allows comparing metrics across different model training runs.
"""

import glob
import os

import matplotlib.colors as mcolors
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from tensorboard.backend.event_processing import event_accumulator

# Configuration
# Dictionary mapping model paths to display names for the legend
MODEL_PATHS = {
    "edgecodedpo/models/dpo_qwen15epochs_longds/logs": "Qwen0.5 / Full dataset",
    "edgecodedpo/models/dpo_gemma_15epochs_fullDS/logs": "Gemma2b / Full dataset",
    "edgecodedpo/models/dpo_output_qwen_15epochs/logs": "Qwen0.5 / Half dataset",
    "edgecodedpo/models/dpo_output_gemma_15epochs/logs": "Gemma2b / Half dataset",
    "edgecodedpo/models/dpo_deepseek_15epochs/logs": "DeepseekCoder1b / Half dataset",
}

# Output directory for plots
OUTPUT_DIR = "comparison_plots"

# Define default metric pairs for visualization
METRIC_PAIRS = {
    "Loss": ["train/loss", "eval/loss"],
    "Gradient Norm": ["train/grad_norm"],
    "Learning Rate": ["train/learning_rate"],
    "Rewards (Chosen)": [
        "train/rewards/chosen",
        "eval/rewards/chosen",
    ],
    "Rewards (Rejected)": [
        "train/rewards/rejected",
        "eval/rewards/rejected",
    ],
    "Reward Accuracies": ["train/rewards/accuracies", "eval/rewards/accuracies"],
    "Reward Margins": ["train/rewards/margins", "eval/rewards/margins"],
    "Log Probabilities": [
        "train/logps/chosen",
        "train/logps/rejected",
        "eval/logps/chosen",
        "eval/logps/rejected",
    ],
    "Logits": [
        "train/logits/chosen",
        "train/logits/rejected",
        "eval/logits/chosen",
        "eval/logits/rejected",
    ],
}

# Plot settings
DPI = 150
SAVE_INDIVIDUAL = True
SAVE_COMBINED = True
FONT_SIZE = 20
LINE_WIDTH = 3


def find_event_files(log_dir: str) -> list[str]:
    """Find all TensorBoard event files in the given directory."""
    pattern = os.path.join(log_dir, "events.out.tfevents.*")
    event_files = glob.glob(pattern)
    if not event_files:
        print(f"Warning: No event files found in {log_dir}")
        return []
    return event_files


def load_data_from_event_file(
    event_file: str, scalar_names: list[str]
) -> dict[str, list[tuple[int, float]]]:
    """Load scalar data from a TensorBoard event file."""
    ea = event_accumulator.EventAccumulator(
        event_file,
        size_guidance={
            event_accumulator.SCALARS: 0,  # Load all scalar events
        },
    )
    ea.Reload()

    available_tags = ea.Tags()["scalars"]
    print(
        f"Available scalar tags in {os.path.basename(event_file)}: {len(available_tags)} tags"
    )

    data = {}
    for name in scalar_names:
        if name in available_tags:
            events = ea.Scalars(name)
            data[name] = [(event.step, event.value) for event in events]
        else:
            print(f"Warning: {name} not found in {event_file}")

    return data


def combine_event_data(
    event_files: list[str], scalar_names: list[str]
) -> dict[str, list[tuple[int, float]]]:
    """Combine data from multiple event files."""
    combined_data = {}

    for event_file in event_files:
        data = load_data_from_event_file(event_file, scalar_names)
        for name, values in data.items():
            if name not in combined_data:
                combined_data[name] = []
            combined_data[name].extend(values)

    # Sort data by step for each metric
    for name in combined_data:
        combined_data[name].sort(key=lambda x: x[0])

    return combined_data


def load_model_data(
    model_paths: dict[str, str], all_metrics: list[str]
) -> dict[str, dict[str, list[tuple[int, float]]]]:
    """Load data for multiple models."""
    model_data = {}

    for model_path, model_name in model_paths.items():
        print(f"\nLoading data for model: {model_name} from {model_path}")
        try:
            event_files = find_event_files(model_path)
            if not event_files:
                continue

            print(f"Found {len(event_files)} event files")
            data = combine_event_data(event_files, all_metrics)

            if data:
                model_data[model_name] = data
            else:
                print(f"No data loaded for model {model_name}")
        except Exception as e:
            print(f"Error loading data for model {model_name}: {e}")

    return model_data


def setup_plot_style():
    """Set up the global plot style with Helvetica font and other settings."""
    # Try to use Helvetica font
    try:
        plt.rcParams["font.family"] = "Helvetica"
    except:
        # If Helvetica is not available, try Arial (similar to Helvetica)
        try:
            plt.rcParams["font.family"] = "Arial"
        except:
            # Fallback to sans-serif if neither is available
            plt.rcParams["font.family"] = "sans-serif"
            plt.rcParams["font.sans-serif"] = ["Helvetica", "Arial"]

    # Set font sizes
    plt.rcParams["font.size"] = FONT_SIZE
    plt.rcParams["axes.titlesize"] = FONT_SIZE
    plt.rcParams["axes.labelsize"] = FONT_SIZE
    plt.rcParams["xtick.labelsize"] = FONT_SIZE
    plt.rcParams["ytick.labelsize"] = FONT_SIZE
    plt.rcParams["legend.fontsize"] = int(FONT_SIZE * 3 / 4)


def create_custom_legend(ax, model_colors, split_styles, show_legend=True):
    """Create a custom legend that separates model colors and split styles."""
    if not show_legend:
        return

    # Create legend for models (colors)
    model_handles = [
        Line2D([0], [0], color=color, lw=LINE_WIDTH) for color in model_colors.values()
    ]
    model_labels = list(model_colors.keys())

    # Create legend for splits (line styles)
    split_handles = []
    split_labels = []

    # Use black color for split style indicators
    for split_name, style in split_styles.items():
        if split_name:  # Skip if empty string
            split_handles.append(
                Line2D([0], [0], color="black", linestyle=style, lw=LINE_WIDTH)
            )
            split_labels.append(split_name)

    # Add model legend on the top right
    if model_handles:
        legend1 = ax.legend(
            model_handles, model_labels, loc="upper right", title="Models"
        )
        ax.add_artist(legend1)

    # Add split legend below model legend
    if split_handles:
        # Calculate position to place below the model legend
        bbox = ax.get_position()
        legend2 = ax.legend(
            split_handles,
            split_labels,
            loc="center right",
            bbox_to_anchor=(0.17, 0.3),
            title="Splits",
        )
        ax.add_artist(legend2)


def create_multi_model_visualizations(
    model_paths: dict[str, str],
    output_dir: str,
    metric_pairs: dict[str, list[str]] | None = None,
    save_individual: bool = True,
    save_combined: bool = True,
    dpi: int = 150,
) -> None:
    """
    Create visualization plots comparing metrics across multiple models.

    Args:
        model_paths: Dictionary mapping model log directories to display names
        output_dir: Directory where to save the generated plots
        metric_pairs: Dictionary mapping plot titles to lists of metrics to include
        save_individual: Whether to save individual plots for each metric group
        save_combined: Whether to save a combined figure with all metrics as subplots
        dpi: Resolution of the output images
    """
    if metric_pairs is None:
        metric_pairs = METRIC_PAIRS

    os.makedirs(output_dir, exist_ok=True)

    # Set up plot style
    setup_plot_style()

    # Flatten the list of metrics to load
    all_metrics = []
    for metrics in metric_pairs.values():
        all_metrics.extend(metrics)

    # Load data for all models
    model_data = load_model_data(model_paths, all_metrics)

    if not model_data:
        print("No data loaded for any model. Exiting.")
        return

    print(f"\nGenerating plots for {len(model_data)} models")

    # Get a colormap for different models
    colors = list(mcolors.TABLEAU_COLORS)
    model_colors = {
        model: colors[i % len(colors)] for i, model in enumerate(model_data.keys())
    }

    # Line styles for train vs eval
    split_styles = {"Train": "-", "Eval": "--"}

    # Create individual plots for each metric group
    if save_individual:
        for title, metrics in metric_pairs.items():
            plt.figure(figsize=(12, 6))
            ax = plt.gca()

            # Check if any model has this metric
            has_data = False
            model_in_plot = set()
            split_in_plot = set()

            for model_name, data in model_data.items():
                model_color = model_colors[model_name]

                for metric in metrics:
                    if data.get(metric):
                        has_data = True
                        steps, values = zip(*data[metric], strict=False)

                        # Determine split (train/eval)
                        if "train/" in metric:
                            split = "Train"
                            style = split_styles["Train"]
                            split_in_plot.add("Train")
                        elif "eval/" in metric:
                            split = "Eval"
                            style = split_styles["Eval"]
                            split_in_plot.add("Eval")
                        else:
                            split = ""
                            style = "-"

                        model_in_plot.add(model_name)

                        # Plot without adding to legend (we'll create custom legend)
                        ax.plot(
                            steps,
                            values,
                            linestyle=style,
                            color=model_color,
                            linewidth=LINE_WIDTH,
                        )

            if has_data:
                # Create filtered dictionaries for models and splits actually in this plot
                plot_model_colors = {
                    model: model_colors[model] for model in model_in_plot
                }
                plot_split_styles = {
                    split: split_styles[split] for split in split_in_plot if split
                }

                # Only show legend on the Loss plot
                show_legend = title == "Loss"
                create_custom_legend(
                    ax, plot_model_colors, plot_split_styles, show_legend
                )

                plt.title(title)
                plt.xlabel("Step")
                plt.ylabel("Value")
                plt.grid(True, alpha=0.3)

                filename = title.replace("/", "_").replace(" ", "_").lower() + ".png"
                plt.savefig(
                    os.path.join(output_dir, filename), dpi=dpi, bbox_inches="tight"
                )
                print(f"Saved {filename}")
            else:
                print(f"No data available for {title}, skipping plot")

            plt.close()

    # Create a combined figure with all subplots
    if save_combined:
        n_plots = len(metric_pairs)
        has_any_data = False

        fig, axs = plt.subplots(n_plots, 1, figsize=(15, 5 * n_plots), squeeze=False)

        for i, (title, metrics) in enumerate(metric_pairs.items()):
            ax = axs[i, 0]
            has_plot_data = False
            model_in_plot = set()
            split_in_plot = set()

            for model_name, data in model_data.items():
                model_color = model_colors[model_name]

                for metric in metrics:
                    if data.get(metric):
                        has_plot_data = True
                        has_any_data = True
                        steps, values = zip(*data[metric], strict=False)

                        # Determine split (train/eval)
                        if "train/" in metric:
                            split = "Train"
                            style = split_styles["Train"]
                            split_in_plot.add("Train")
                        elif "eval/" in metric:
                            split = "Eval"
                            style = split_styles["Eval"]
                            split_in_plot.add("Eval")
                        else:
                            split = ""
                            style = "-"

                        model_in_plot.add(model_name)

                        # Plot without adding to legend (we'll create custom legend)
                        ax.plot(
                            steps,
                            values,
                            linestyle=style,
                            color=model_color,
                            linewidth=LINE_WIDTH,
                        )

            if has_plot_data:
                # Create filtered dictionaries for models and splits actually in this plot
                plot_model_colors = {
                    model: model_colors[model] for model in model_in_plot
                }
                plot_split_styles = {
                    split: split_styles[split] for split in split_in_plot if split
                }

                # Only show legend on the Loss plot
                show_legend = title == "Loss"
                create_custom_legend(
                    ax, plot_model_colors, plot_split_styles, show_legend
                )

                ax.set_title(title)
                ax.set_xlabel("Step")
                ax.set_ylabel("Value")
                ax.grid(True, alpha=0.3)
            else:
                ax.set_title(f"{title} - No data available")

        if has_any_data:
            # Adjust layout to make room for legends
            plt.tight_layout()
            # Add extra space between subplots for legends
            plt.subplots_adjust(hspace=0.4)
            plt.savefig(
                os.path.join(output_dir, "combined_metrics.png"),
                dpi=dpi * 1.2,
                bbox_inches="tight",
            )
            print("Saved combined_metrics.png")
        else:
            print("No data available for any metrics, skipping combined plot")

        plt.close()

    print(f"All plots saved to {output_dir}")


if __name__ == "__main__":
    # You can modify these variables directly in the script instead of using argparse
    create_multi_model_visualizations(
        model_paths=MODEL_PATHS,
        output_dir=OUTPUT_DIR,
        metric_pairs=METRIC_PAIRS,
        save_individual=SAVE_INDIVIDUAL,
        save_combined=SAVE_COMBINED,
        dpi=DPI,
    )
