# EdgeCodeDPO

A toolkit for generating and fine-tuning code models using Direct Preference Optimization (DPO).

## Project Status

⚠️ **EARLY DEVELOPMENT** ⚠️

This project is in its very early stages. Currently, only the data generation component (v0) has been implemented. Many features are planned but not yet available.

## Overview

EdgeCodeDPO is designed to create high-quality datasets for training and fine-tuning code generation models. The project generates pairs of code examples (chosen and rejected) across various programming domains, tasks, and code structures to enable effective preference optimization for code models.

## Current Features

- **Dataset Generator**: Creates pairs of code examples using a two-stage pipeline:
  - Stage 1: Generates basic, functional code examples across various domains
  - Stage 2: Creates improved versions with proper type annotations, documentation, and best practices
- **Configuration-based**: Domains, tasks, libraries, and code forms are defined in YAML config files
- **Asynchronous Processing**: Efficiently processes multiple API requests concurrently
- **HuggingFace Dataset Integration**: Outputs are saved in HuggingFace dataset format
- **Typer CLI Interface**: Easy-to-use command-line interface for all operations

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/EdgeCodeDPO.git
cd EdgeCodeDPO

# Development installation with all dependencies
pip install -e ".[all]"

# Or minimal installation 
pip install -e .

# Set up your OpenAI API key
export OPENAI_API_KEY="your-api-key-here"
```

Alternatively, you can create a `.env` file in the `edgecodedpo` directory with:

```
OPENAI_KEY=your-api-key-here
```

## Usage

### CLI Commands

EdgeCodeDPO provides a command-line interface with the following commands:

```bash
# Show available commands
edgecodedpo --help

# Generate a dataset
edgecodedpo generate --config edgecodedpo/configs/dataset.yaml --output data/gen_data

# Check version
edgecodedpo version
```

### Generating a Dataset

```bash
edgecodedpo generate --config edgecodedpo/configs/dataset.yaml --output data/gen_data --samples 10 --model gpt-4o-mini
```

#### Parameters

- `--config`, `-c`: Path to the configuration file (default: `edgecodedpo/configs/dataset.yaml`)
- `--output`, `-o`: Path to save the generated dataset (default: `edgecodedpo/data/gen_data`)
- `--samples`, `-s`: Number of combinations to sample (default: all combinations)
- `--batch-size`, `-b`: Number of concurrent API requests (default: 5)
- `--model`, `-m`: OpenAI model to use (default: `gpt-4o-mini`)
- `--system-message`: Optional system message for the API
- `--no-intermediate`: Don't save intermediate results

### Configuration File

The `edgecodedpo/configs/dataset.yaml` file defines the domains, tasks, libraries, and code forms used for dataset generation. You can customize this file to generate examples for specific domains or tasks.

## Project Structure

```
EdgeCodeDPO/
├── edgecodedpo/
│   ├── __main__.py          # CLI entry point
│   ├── config.py            # Project configuration
│   ├── clients/
│   │   └── openai_client.py # Async client for OpenAI API
│   ├── configs/
│   │   └── dataset.yaml     # Configuration for dataset generation
│   └── data/
│       ├── dataset_generator.py  # Main dataset generation script
│       └── prompt_generator.py   # Prompt creation utilities
├── pyproject.toml           # Package configuration
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

## Roadmap

- [x] Initial dataset generation pipeline
- [x] CLI interface implementation
- [ ] Data quality evaluation and filtering
- [ ] Direct Preference Optimization (DPO) training
- [ ] Model evaluation framework
- [ ] Web interface for dataset exploration
- [ ] Support for additional model providers
- [ ] Edge deployment optimizations

## Dependencies

Main dependencies:
- OpenAI API (>= 1.0.0)
- HuggingFace Datasets (>= 2.14.0)
- Typer (for CLI)
- Pandas (>= 2.0.0)
- PyYAML
- Pydantic Settings
- AsyncIO and AIOHTTP

Development dependencies are available through the `[dev]` extra.

## Contributing

As this project is in early development, contributions are welcome but the codebase may change significantly. Please reach out before investing significant time in contributions.

## License

[MIT License](LICENSE) (License file not yet included)