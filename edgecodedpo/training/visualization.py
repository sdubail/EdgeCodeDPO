"""
Visualization module for extracting and plotting TensorBoard metrics.
"""

import glob
import os

import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

# define default metric pairs for visualization
METRIC_PAIRS = {
    "Loss": ["train/loss", "eval/loss"],
    "Gradient Norm": ["train/grad_norm"],
    "Learning Rate": ["train/learning_rate"],
    "Rewards": [
        "train/rewards/chosen",
        "train/rewards/rejected",
        "eval/rewards/chosen",
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


def find_event_files(log_dir: str) -> list[str]:
    """Find all TensorBoard event files in the given directory."""
    pattern = os.path.join(log_dir, "events.out.tfevents.*")
    event_files = glob.glob(pattern)
    if not event_files:
        raise FileNotFoundError(f"No event files found in {log_dir}")
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


def create_visualizations(
    log_dir: str,
    output_dir: str,
    metric_pairs: dict[str, list[str]] | None = None,
    save_individual: bool = True,
    save_combined: bool = True,
    dpi: int = 150,
) -> None:
    """
    Create visualization plots for training metrics.

    Args:
        log_dir: Directory containing TensorBoard event files
        output_dir: Directory where to save the generated plots
        metric_pairs: Dictionary mapping plot titles to lists of metrics to include
                     (defaults to the module's METRIC_PAIRS if None)
        save_individual: Whether to save individual plots for each metric group
        save_combined: Whether to save a combined figure with all metrics as subplots
        dpi: Resolution of the output images
    """
    if metric_pairs is None:
        metric_pairs = METRIC_PAIRS

    os.makedirs(output_dir, exist_ok=True)

    all_metrics = []
    for metrics in metric_pairs.values():
        all_metrics.extend(metrics)

    try:
        event_files = find_event_files(log_dir)
        print(f"Found {len(event_files)} event files")

        data = combine_event_data(event_files, all_metrics)

        # Create individual plots
        if save_individual:
            for title, metrics in metric_pairs.items():
                plt.figure(figsize=(12, 6))

                for metric in metrics:
                    if data.get(metric):
                        steps, values = zip(*data[metric], strict=False)
                        label = metric.split("/")[-1] if "/" in metric else metric
                        if "train/" in metric:
                            plt.plot(
                                steps, values, label=f"Train {label}", linestyle="-"
                            )
                        elif "eval/" in metric:
                            plt.plot(
                                steps, values, label=f"Eval {label}", linestyle="--"
                            )
                        else:
                            plt.plot(steps, values, label=label)

                plt.title(title)
                plt.xlabel("Step")
                plt.ylabel("Value")
                plt.legend()
                plt.grid(True, alpha=0.3)

                filename = title.replace("/", "_").replace(" ", "_").lower() + ".png"
                plt.savefig(
                    os.path.join(output_dir, filename), dpi=dpi, bbox_inches="tight"
                )
                print(f"Saved {filename}")
                plt.close()

        # Create a combined figure with all subplots
        if save_combined:
            n_plots = len(metric_pairs)
            fig, axs = plt.subplots(
                n_plots, 1, figsize=(15, 5 * n_plots), squeeze=False
            )

            for i, (title, metrics) in enumerate(metric_pairs.items()):
                ax = axs[i, 0]

                for metric in metrics:
                    if data.get(metric):
                        steps, values = zip(*data[metric], strict=False)
                        label = metric.split("/")[-1] if "/" in metric else metric
                        if "train/" in metric:
                            ax.plot(
                                steps, values, label=f"Train {label}", linestyle="-"
                            )
                        elif "eval/" in metric:
                            ax.plot(
                                steps, values, label=f"Eval {label}", linestyle="--"
                            )
                        else:
                            ax.plot(steps, values, label=label)

                ax.set_title(title)
                ax.set_xlabel("Step")
                ax.set_ylabel("Value")
                ax.legend()
                ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "combined_metrics.png"), dpi=dpi * 1.2)
            print("Saved combined_metrics.png")
            plt.close()

        print(f"All plots saved to {output_dir}")

    except Exception as e:
        print(f"Error generating visualizations: {e}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Visualize TensorBoard metrics from event files"
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="edgecodedpo/models/dpo",
        help="Directory containing TensorBoard event files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="edgecodedpo/metric_plots",
        help="Output directory for plot images",
    )

    args = parser.parse_args()
    create_visualizations(args.log_dir, args.output_dir)


if __name__ == "__main__":
    main()
