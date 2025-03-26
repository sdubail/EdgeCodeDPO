import json
import os
from typing import Any, Dict, List

import seaborn as sns
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from datasets import Dataset, load_dataset, load_from_disk

from edgecodedpo.training.eval_metrics import (
    evaluate_code_quality,
    execute_code,
)
from edgecodedpo.utils.generated_code_parsing import (
    assemble_code_blocks,
    extract_code_blocks,
    preprocess_code_blocks,
)

def save_plot(fig, output_dir, filename):
    filepath = os.path.join(output_dir, filename)
    fig.savefig(filepath, bbox_inches="tight")
    plt.close(fig)
    
def load_and_evaluate_dataset(
    dataset_path: str,
    output_dir: str,
    dataset_split: str,
    num_examples: int = 10,
    batch_size: int = 4,
) -> None:
    """
    Compute metrics and create figures to visualize them through different subgroups

    Args:
        dataset_path: Path to the dataset
        output_dir: Directory to save the evaluation results
        dataset_split: Split of the dataset the metrics will be computed on
        num_examples: Number of examples to evaluate
        batch_size: Number of examples to process in a single batch
    """
    # Create output directory if it doesn't exist
    output_dir = os.path.join(output_dir, dataset_path.split('/')[-1], dataset_split)
    os.makedirs(output_dir, exist_ok=True)
    
    if os.path.exists(os.path.join(output_dir, "dataset_eval_results.json")):
        print("Metrics have already been computed for this dataset, going directly to figure generation")
    else:
        compute_dataset_metrics(
                dataset_path = dataset_path,
                output_dir = output_dir,
                dataset_split = dataset_split,
                num_examples = num_examples,
                batch_size = batch_size,
        )
    print(f"Making figures from metrics registered in {os.path.join(output_dir, 'dataset_eval_results.json')}")
    create_metrics_figures(
        json_path=os.path.join(output_dir, 'dataset_eval_results.json'),
        output_dir=output_dir
    )

def compute_dataset_metrics(
    dataset_path: str,
    output_dir: str,
    dataset_split: str,
    num_examples: int = 10,
    batch_size: int = 4,
    ) -> None:
    """
    Load a dataset and evaluate it by comparing chosen vs rejected code and subcategorizing the analysis.
    """
    # Load dataset
    if os.path.exists(dataset_path):
        print(f"Loading dataset from local path: {dataset_path}")
        dataset = load_from_disk(dataset_path)
    else:
        print(f"Attempting to load dataset from HuggingFace Hub: {dataset_path}")
        try:
            if ":" in dataset_path:
                repo_id, split = dataset_path.split(":", 1)
                dataset = load_dataset(repo_id, split=split)
            else:
                dataset = load_dataset(dataset_path)
        except Exception as e:
            raise ValueError(
                f"Failed to load dataset from HuggingFace Hub: {dataset_path}. Error: {e!s}"
            )

    # Select evaluation examples
    try:
        eval_dataset = dataset[dataset_split].select(
            range(min(num_examples, dataset[dataset_split].num_rows))
        )
    except Exception as e:
        raise ValueError(f"{dataset_split} not found in dataset {dataset_path}. Only available dataset_split are : {eval_dataset.keys()}")

    if not ("chosen" in eval_dataset[0].keys() or "rejected" in eval_dataset[0].keys()):
        raise ValueError("Dataset is expected to have columns 'chosen' and 'rejected'.")

    # Process dataset in batches
    results = []
    for batch_start in tqdm(
        range(0, min(num_examples, len(eval_dataset)), batch_size),
        total = int(min(num_examples, len(eval_dataset)) / batch_size),
        desc = "Computing metrics"):
        
        batch_end = min(batch_start + batch_size, len(eval_dataset))
        batch = eval_dataset[batch_start:batch_end]
        
        # Process each example in the batch
        for i in range(batch_end - batch_start):
            chosen = batch["chosen"][i]
            rejected = batch["rejected"][i]
            
            # Evaluate chosen code
            if chosen:
                chosen_code = chosen[0]["content"]
                chosen_blocks = extract_code_blocks(chosen_code)
                chosen_full_script = assemble_code_blocks(chosen_blocks)
                chosen_preprocess_blocks = preprocess_code_blocks(chosen_blocks)

                if len(chosen_preprocess_blocks) > 0:
                    chosen_metrics = evaluate_code_quality(chosen_preprocess_blocks[0])
                    chosen_metrics["execution_result"] = execute_code(chosen_full_script)

            # Evaluate rejected code
            if rejected:
                rejected_code = rejected[0]["content"]
                rejected_blocks = extract_code_blocks(rejected_code)
                rejected_full_script = assemble_code_blocks(rejected_blocks)
                rejected_preprocess_blocks = preprocess_code_blocks(rejected_blocks)

                if len(rejected_preprocess_blocks) > 0:
                    rejected_metrics = evaluate_code_quality(rejected_preprocess_blocks[0])
                    rejected_metrics["execution_result"] = execute_code(rejected_full_script)
            
            # Save the results
            result = {}
            for key, value in batch.items():
                if not key in ["prompt", "chosen", "rejected"]:
                    result[key] = value[i]
            result["prompt"] = batch["prompt"][i][0]
            result["chosen_code"] = chosen_code if chosen else None
            result["rejected_code"] = rejected_code if rejected else None
            result["chosen_metrics"] = chosen_metrics if chosen else None
            result["rejected_metrics"] = rejected_metrics if rejected else None
            
            results.append(result)

    # Convert all results and metrics to JSON-serializable format
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.generic):
            return obj.item()
        elif isinstance(obj, dict):
            return {key: convert_numpy(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        return obj

    results = convert_numpy(results)

    # Save results to a file
    with open(os.path.join(output_dir, "dataset_eval_results.json"), "w") as f:
        json.dump(
            results,
            f,
            indent=2,
        )

    # Print summary of evaluation results
    print(
        f"Evaluation completed. Results saved to: {os.path.join(output_dir, 'dataset_eval_results.json')}"
    )

def create_metrics_figures(
    json_path: str,
    output_dir: str,
) -> None:
    """
    From json metrics computation summary, create visualization figures.
    """
    
    with open(json_path, "r") as f:
        results = json.load(f)
    
    metrics = results[0]["chosen_metrics"].keys()
    print(f"Metrics which will be aggregated: {metrics}")
    categories = set(results[0].keys()) - {"chosen_code", "rejected_code", "prompt", "rejected_metrics", "chosen_metrics"}
    print(f"Explored categories: {categories}")
    
    aggregated_metrics = {"chosen": defaultdict(list), "rejected": defaultdict(list)}
    aggregated_metrics_category = {
        cat: defaultdict(lambda: {"chosen": defaultdict(list), "rejected": defaultdict(list)})
        for cat in categories
    }
    
    for res in tqdm(results, total=len(results), desc="Aggregating metrics"):
        for metric in metrics:
            if metric == "execution_result":
                res["chosen_metrics"][metric] = 1 if res["chosen_metrics"][metric]["success"] else 0
                res["rejected_metrics"][metric] = 1 if res["rejected_metrics"][metric]["success"] else 0
            if not res["chosen_metrics"].get(metric) is None:
                aggregated_metrics["chosen"][metric].append(res["chosen_metrics"][metric])
                for cat in categories:
                    aggregated_metrics_category[cat][res[cat]]["chosen"][metric].append(res["chosen_metrics"][metric])
            if not res["rejected_metrics"].get(metric) is None:
                aggregated_metrics["rejected"][metric].append(res["rejected_metrics"][metric])
                for cat in categories:
                    aggregated_metrics_category[cat][res[cat]]["rejected"][metric].append(res["rejected_metrics"][metric])
    
    sns.set(style="whitegrid")
    
    for metric in metrics:
        if aggregated_metrics["chosen"].get(metric) and aggregated_metrics["rejected"].get(metric):
            data = aggregated_metrics["chosen"][metric] + aggregated_metrics["rejected"][metric]
            types = ["Chosen"] * len(aggregated_metrics["chosen"][metric]) + ["Rejected"] * len(aggregated_metrics["rejected"][metric])
            df = {"Type": types, "Value": data}
            
            plt.figure(figsize=(8, 6))
            sns.boxplot(x="Type", y="Value", data=df, palette=["#1f77b4", "#ff7f0e"], showfliers=False)
            plt.title(f"Comparison of {metric} (Box Plot)")
            plt.ylabel(metric)
            save_plot(plt.gcf(), output_dir, f"metric_{metric}_boxplot.png")
            
            plt.figure(figsize=(8, 6))
            sns.barplot(x=["Chosen", "Rejected"], y=[np.mean(aggregated_metrics["chosen"][metric]), np.mean(aggregated_metrics["rejected"][metric])], palette=["#1f77b4", "#ff7f0e"])
            plt.title(f"Comparison of {metric} (Bar Plot)")
            plt.ylabel(metric)
            save_plot(plt.gcf(), output_dir, f"metric_{metric}_barplot.png")
    
    for category in categories:
        category_path = os.path.join(output_dir, f"metrics_across_{category}")
        os.makedirs(category_path, exist_ok=True)
        
        for metric in metrics:
            data = []
            data_mean = []
            categories_list = []
            categories_list_mean = []
            types = []
            types_mean = []
            for cat_name, cat_metrics in aggregated_metrics_category[category].items():
                if metric in cat_metrics["chosen"] and metric in cat_metrics["rejected"]:
                    data.extend(cat_metrics["chosen"][metric])
                    data_mean.append(np.mean(cat_metrics["chosen"][metric]))
                    data.extend(cat_metrics["rejected"][metric])
                    data_mean.append(np.mean(cat_metrics["rejected"][metric]))
                    categories_list.extend([cat_name] * (len(cat_metrics["chosen"][metric]) + len(cat_metrics["rejected"][metric])))
                    categories_list_mean.extend([cat_name,cat_name])
                    types.extend(["Chosen"] * len(cat_metrics["chosen"][metric]) + ["Rejected"] * len(cat_metrics["rejected"][metric]))
                    types_mean.extend(["Chosen", "Rejected"])
            
            if data:
                df = {"Category": categories_list, "Value": data, "Type": types}
                df_mean = {"Category": categories_list_mean, "Value": data_mean, "Type": types_mean}
                
                plt.figure(figsize=(12, 6))
                sns.boxplot(x="Category", y="Value", hue="Type", data=df, palette=["#1f77b4", "#ff7f0e"], showfliers=False)
                plt.xticks(rotation=45, ha="right")
                plt.title(f"{metric} distributions across {category} (Box Plot)")
                plt.ylabel(metric)
                plt.legend(title="Type")
                save_plot(plt.gcf(), category_path, f"{category}_{metric}_boxplot.png")
                
                plt.figure(figsize=(12, 6))
                sns.barplot(x="Category", y=data_mean, hue=types_mean, data=df_mean, palette=["#1f77b4", "#ff7f0e"])
                plt.xticks(rotation=45, ha="right")
                plt.title(f"{metric} distributions across {category} (Bar Plot)")
                plt.ylabel(metric)
                plt.legend(title="Type")
                save_plot(plt.gcf(), category_path, f"{category}_{metric}_barplot.png")
    
    print("Visualization completed and saved in output directory.")
