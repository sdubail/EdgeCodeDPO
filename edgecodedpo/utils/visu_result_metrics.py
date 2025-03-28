"""
Visualization module for creating grouped bar charts comparing evaluation metrics
across different models and prompt types.
"""

import json
import os

import matplotlib.colors as mcolors
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

MODEL_PATHS = {
    "edgecodedpo/models/evaluation/eval_results_qwen_15epochs_fullDS_full_test_promptboost.json": {
        "name": "Qwen0.5B-dpo + prompt eng.",
        "type": "fine-tuned",
    },
    "edgecodedpo/models/evaluation/eval_results_qwen_full_test_promptboost.json": {
        "name": "Qwen0.5B-base + prompt eng.",
        "type": "base",
    },
    # "edgecodedpo/models/evaluation/eval_results_qwen15epochs_fullDS_full_test copy.json": {
    #     "name": "Qwen0.5B-dpo",
    #     "type": "fine-tuned",
    # },
    # "edgecodedpo/models/evaluation/eval_results_qwen_full_test.json": {
    #     "name": "Qwen0.5B-base",
    #     "type": "base",
    # },
    # "edgecodedpo/models/evaluation/eval_results_gemma15epochs_fullDS_full_test.json": {
    #     "name": "Gemma2B-dpo",
    #     "type": "fine-tuned",
    # },
    # "edgecodedpo/models/evaluation/eval_results_gemma_full_test.json": {
    #     "name": "Gemma2B-base",
    #     "type": "base",
    # },
}
OUTPUT_DIR = "evaluation_plots"

METRICS = [
    "type_annotation_coverage",
    "comment_density",
    "docstring_coverage",
    "code_complexity",
    "execution_success_rate",
    "similarity_to_chosen",
    "similarity_to_rejected",
]

METRIC_TITLES = {
    "type_annotation_coverage": "Type Annotation Coverage",
    "comment_density": "Comment Density",
    "docstring_coverage": "Docstring Coverage",
    "code_complexity": "Code Complexity",
    "pep8_compliance": "PEP8 Compliance",
    "execution_success_rate": "Execution Success Rate",
    "similarity_to_chosen": "Similarity to Chosen",
    "similarity_to_rejected": "Similarity to Rejected",
}

PROMPT_TYPES = ["default", "code_form", "code_form_types"]

DPI = 150
FONT_SIZE = 25
LINE_WIDTH = 3
BAR_WIDTH = 0.4
GROUPS_SPACING = 0.4


def setup_plot_style():
    """Set up the global plot style with Helvetica font and other settings."""
    try:
        plt.rcParams["font.family"] = "Helvetica"
    except:
        try:
            plt.rcParams["font.family"] = "Arial"
        except:
            plt.rcParams["font.family"] = "sans-serif"
            plt.rcParams["font.sans-serif"] = ["Helvetica", "Arial"]

    plt.rcParams["font.size"] = FONT_SIZE
    plt.rcParams["axes.titlesize"] = FONT_SIZE
    plt.rcParams["axes.labelsize"] = FONT_SIZE
    plt.rcParams["xtick.labelsize"] = FONT_SIZE
    plt.rcParams["ytick.labelsize"] = FONT_SIZE
    plt.rcParams["legend.fontsize"] = FONT_SIZE


def load_model_data(model_paths):
    """
    Load evaluation results from JSON files.

    Args:
        model_paths: Dictionary mapping model paths to model info

    Returns:
        Dictionary with model data
    """
    model_data = {}

    for model_path, model_info in model_paths.items():
        model_name = model_info["name"]
        print(f"\nLoading data for model: {model_name} from {model_path}")

        try:
            with open(model_path) as f:
                data = json.load(f)

            if "metrics" in data:
                model_data[model_name] = {
                    "type": model_info["type"],
                    "global_metrics": data["metrics"],
                    "prompt_metrics": data["prompt_type_metrics"]
                    if "prompt_type_metrics" in data
                    else {},
                }
                print(f"Successfully loaded data for {model_name}")
            else:
                print(
                    f"Error: JSON file for {model_name} does not contain 'metrics' key"
                )
        except Exception as e:
            print(f"Error loading data for model {model_name}: {e}")

    return model_data


def create_grouped_bar_charts(model_data, metrics, prompt_types, output_dir):
    """
    Create a single figure with multiple subplots, one for each metric.

    Args:
        model_data: Dictionary containing model metrics
        metrics: List of metrics to visualize
        prompt_types: List of prompt types to compare
        output_dir: Directory to save the output plots
    """
    os.makedirs(output_dir, exist_ok=True)

    base_models = {
        name: data for name, data in model_data.items() if data["type"] == "base"
    }
    finetuned_models = {
        name: data for name, data in model_data.items() if data["type"] == "fine-tuned"
    }

    base_color = "royalblue"
    finetuned_color = "firebrick"

    n_prompt_types = len(prompt_types)
    n_models = len(model_data)
    x = np.arange(n_prompt_types)

    valid_metrics = []
    for metric in metrics:
        skip_metric = True
        for model_name, model_info in model_data.items():
            for prompt_type in prompt_types:
                if prompt_type in model_info["prompt_metrics"]:
                    value = model_info["prompt_metrics"][prompt_type].get(metric)
                    if value is not None:
                        skip_metric = False
                        break
            if not skip_metric:
                break

        if not skip_metric:
            valid_metrics.append(metric)
        else:
            print(f"Skipping {metric} as it contains only null values")

    if not valid_metrics:
        print("No valid metrics to plot. Exiting.")
        return

    n_metrics = len(valid_metrics)
    n_cols = 2
    n_rows = (n_metrics + n_cols - 1) // n_cols

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(18, 6 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axs = np.array([[axs]])
    elif n_rows == 1:
        axs = np.array([axs])
    elif n_cols == 1:
        axs = np.array([[ax] for ax in axs])

    for idx, metric in enumerate(valid_metrics):
        row = idx // n_cols
        col = idx % n_cols
        ax = axs[row, col]

        bar_positions = []
        bar_heights = []
        bar_colors = []
        bar_labels = []

        for i, (model_name, model_info) in enumerate(base_models.items()):
            for j, prompt_type in enumerate(prompt_types):
                if prompt_type in model_info["prompt_metrics"]:
                    value = model_info["prompt_metrics"][prompt_type].get(metric)
                    if value is not None:
                        position = (
                            x[j] - BAR_WIDTH * (n_models / 2 - 0.5) + i * BAR_WIDTH
                        )

                        bar_positions.append(position)
                        bar_heights.append(value)
                        bar_colors.append(base_color)
                        bar_labels.append(model_name)

        for i, (model_name, model_info) in enumerate(finetuned_models.items()):
            for j, prompt_type in enumerate(prompt_types):
                if prompt_type in model_info["prompt_metrics"]:
                    value = model_info["prompt_metrics"][prompt_type].get(metric)
                    if value is not None:
                        position = (
                            x[j]
                            - BAR_WIDTH * (n_models / 2 - 0.5)
                            + (i + len(base_models)) * BAR_WIDTH
                        )

                        bar_positions.append(position)
                        bar_heights.append(value)
                        bar_colors.append(finetuned_color)
                        bar_labels.append(model_name)

        ax.bar(
            bar_positions,
            bar_heights,
            width=BAR_WIDTH,
            color=bar_colors,
            linewidth=1,
            edgecolor="black",
        )

        metric_title = METRIC_TITLES.get(metric, metric)
        ax.set_title(metric_title)
        ax.set_xlabel("Prompt Type")
        ax.set_ylabel("Value")
        if metric == "docstring_coverage":
            ax.set_ylim(0, 1.0)
        elif metric == "type_annotation_coverage":
            ax.set_ylim(0, 0.5)
        elif metric == "comment_density":
            ax.set_ylim(0, 0.25)
        elif "similarity" in metric:
            ax.set_ylim(0.98, 1.0)

        ax.set_xticks(x)
        ax.set_xticklabels(prompt_types)

        if idx == 0:
            legend_elements = []

            for model_name in model_data.keys():
                if model_name in base_models:
                    color = base_color
                else:
                    color = finetuned_color
                legend_elements.append(
                    Line2D(
                        [0],
                        [0],
                        color=color,
                        lw=0,
                        marker="s",
                        markersize=10,
                        label=model_name,
                    )
                )

            ax.legend(
                handles=legend_elements,
                loc="upper left",
            )

        ax.grid(True, alpha=0.3, axis="y")

    for idx in range(len(valid_metrics), n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axs[row, col].axis("off")

    plt.tight_layout()

    filename = "all_metrics_comparison.png"
    plt.savefig(os.path.join(output_dir, filename), dpi=DPI, bbox_inches="tight")
    print(f"Saved {filename}")
    plt.close()


def create_evaluation_visualizations(
    model_paths, output_dir, metrics=None, prompt_types=None
):
    """
    Main function to create evaluation visualizations.

    Args:
        model_paths: Dictionary mapping model paths to info
        output_dir: Directory to save the plots
        metrics: List of metrics to visualize (uses default if None)
        prompt_types: List of prompt types to compare (uses default if None)
    """
    if metrics is None:
        metrics = METRICS

    if prompt_types is None:
        prompt_types = PROMPT_TYPES

    setup_plot_style()

    model_data = load_model_data(model_paths)

    if not model_data:
        print("No data loaded for any model. Exiting.")
        return

    print(f"\nGenerating plots for {len(model_data)} models")

    create_grouped_bar_charts(model_data, metrics, prompt_types, output_dir)

    print(f"All plots saved to {output_dir}")


if __name__ == "__main__":
    create_evaluation_visualizations(model_paths=MODEL_PATHS, output_dir=OUTPUT_DIR)
