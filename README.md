# EdgeCodeDPO

A toolkit for generating and fine-tuning code models using Direct Preference Optimization (DPO).

## Project Status

⚠️ **EARLY DEVELOPMENT** ⚠️

This project is in its early stages with a focus on high-quality data generation and DPO training. Basic functionality for both components is now implemented, with more features in development.

## Overview

EdgeCodeDPO is designed to create high-quality datasets for training and fine-tuning code generation models with DPO, a more efficient technique for aligning models with human preferences compared to traditional RLHF. The project generates pairs of code examples (chosen and rejected) across various programming domains, tasks, and code structures, and then provides tools to train models using these preferences.

## Current Features

### Dataset Generation

- **Two-Stage Pipeline**: Creates pairs of code examples:
  - Stage 1: Generates basic, functional code examples
  - Stage 2: Creates improved versions with proper type annotations, documentation, and best practices
- **Comprehensive Domain Coverage**: 16+ programming domains with diverse tasks and code structures
- **Configuration-based**: Domains, tasks, libraries, and code forms defined in YAML files
- **Asynchronous Processing**: Efficiently processes multiple API requests concurrently
- **Dataset Analysis**: Tools to analyze token length statistics and distribution
- **HuggingFace Integration**: Outputs saved in HuggingFace dataset format with push-to-hub support

### DPO Training

- **Direct Preference Optimization**: Fine-tune models using DPO, a more efficient alternative to RLHF
- **Quantization Support**: Train with 4-bit quantization for memory efficiency
- **LoRA Support**: Efficient fine-tuning with Low-Rank Adaptation
- **Flexible Configuration**: YAML-based configuration for training parameters
- **HuggingFace Integration**: Seamless integration with HuggingFace models and libraries
- **Model Evaluation**: Tools to evaluate fine-tuned models

### CLI Interface

- **Comprehensive Commands**: Easy-to-use command-line interface for all operations
- **Data Generation**: Generate datasets with customizable parameters
- **Dataset Statistics**: Analyze token length distributions and other metrics
- **DPO Training**: Train models with various optimization methods
- **Model Evaluation**: Evaluate trained models on test examples

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/EdgeCodeDPO.git
cd EdgeCodeDPO

# Development installation with all dependencies
pip install -e ".[all]"

# Or minimal installation 
pip install -e .

# Set up API keys
export OPENAI_API_KEY="your-openai-api-key"
export HF_TOKEN="your-huggingface-token"  # Required for pushing to HuggingFace Hub
```

Alternatively, you can create a `.env` file in the `edgecodedpo` directory with:

```
OPENAI_KEY=your-openai-api-key
HF_KEY=your-huggingface-token
```

## Usage

### Generating a Dataset

```bash
# Generate a dataset with default settings
edgecodedpo generate

# Generate with custom settings
edgecodedpo generate --config edgecodedpo/configs/dataset.yaml --output data/gen_data --samples 10 --model gpt-4o-mini --batch-size 5
```

#### Parameters

- `--config`, `-c`: Path to the configuration file (default: `edgecodedpo/configs/dataset.yaml`)
- `--output`, `-o`: Path to save the generated dataset (default: `edgecodedpo/data/gen_data`)
- `--samples`, `-s`: Number of combinations to sample (default: all combinations)
- `--batch-size`, `-b`: Number of concurrent API requests (default: 5)
- `--model`, `-m`: OpenAI model to use (default: `gpt-4o-mini`)
- `--system-message`: Optional system message for the API
- `--no-intermediate`: Don't save intermediate results

### Analyzing Token Length Statistics

```bash
# Generate token length statistics
edgecodedpo stats --dataset simondubail/edgecodedpo --tokenizer Qwen/Qwen2-0.5B-Instruct --output edgecodedpo/data/stats
```

### Uploading to HuggingFace Hub

```bash
# Upload a dataset to HuggingFace Hub
edgecodedpo upload --dataset-path edgecodedpo/data/gen_data/dataset --repo-id yourusername/edgecodedpo
```

### Training with DPO

```bash
# Train a model using DPO
edgecodedpo train train --model Qwen/Qwen2-0.5B-Instruct --dataset edgecodedpo/data/gen_data/dataset --output edgecodedpo/models/dpo --config edgecodedpo/configs/training.yaml
```

#### Training Parameters

- `--model`, `-m`: Base model to use
- `--dataset`, `-d`: Path to the dataset
- `--output`, `-o`: Directory to save the trained model
- `--config`, `-c`: Path to the training configuration file
- `--epochs`, `-e`: Number of training epochs
- `--learning-rate`, `-lr`: Learning rate
- `--batch-size`, `-b`: Per-device training batch size
- `--beta`: DPO beta parameter
- `--loss-type`: DPO loss type (sigmoid, hinge, ipo, etc.)
- `--push-to-hub`: Push model to HuggingFace Hub
- `--hub-model-id`: HuggingFace Hub model ID

### Evaluating a Trained Model

```bash
# Evaluate a trained model
edgecodedpo train evaluate --model edgecodedpo/models/dpo --dataset edgecodedpo/data/gen_data/dataset --output edgecodedpo/models/evaluation
```

## Configuration Files

### Dataset Configuration

The `edgecodedpo/configs/dataset.yaml` file defines domains, tasks, libraries, and code forms used for dataset generation. You can customize this file to generate examples for specific domains or tasks.

### Training Configuration

The `edgecodedpo/configs/training.yaml` file defines settings for DPO training, including model parameters, optimization settings, and training hyperparameters.

## Project Structure

```
EdgeCodeDPO/
├── edgecodedpo/
│   ├── __main__.py          # CLI entry point
│   ├── config.py            # Project configuration
│   ├── clients/
│   │   └── openai_client.py # Async client for OpenAI API
│   ├── configs/
│   │   ├── dataset.yaml     # Configuration for dataset generation
│   │   └── training.yaml    # Configuration for DPO training
│   ├── data/
│   │   ├── dataset_generator.py  # Dataset generation script
│   │   └── prompt_generator.py   # Prompt creation utilities
│   └── training/
│       ├── cli.py           # Training CLI commands
│       ├── dpo.py           # DPO training functionality
│       └── integration.py   # Integration with main CLI
├── pyproject.toml           # Package configuration
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

## Roadmap

- [x] Initial dataset generation pipeline
- [x] CLI interface implementation
- [x] Dataset statistics and analysis tools
- [x] Direct Preference Optimization (DPO) training
- [ ] Advanced data quality evaluation and filtering
- [ ] Model evaluation framework improvements
- [ ] Web interface for dataset exploration
- [ ] Support for additional model providers
- [ ] Edge deployment optimizations

## Dependencies

Main dependencies:
- OpenAI API (>= 1.0.0)
- HuggingFace Datasets (>= 2.14.0)
- HuggingFace Transformers (>= 4.38.0)
- TRL (>= 0.8.1) for DPO training
- PEFT (>= 0.7.0) for LoRA
- Typer (for CLI)
- Pandas (>= 2.0.0)
- PyYAML
- Pydantic Settings
- AsyncIO and AIOHTTP

Development dependencies are available through the `[dev]` extra.

## Contributing

As this project is in early development, contributions are welcome but the codebase may change significantly. Please reach out before investing significant time in contributions.

## License

[MIT License](LICENSE)