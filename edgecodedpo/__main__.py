def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a code dataset using OpenAI API"
    )
    parser.add_argument(
        "--config",
        default="configs/dataset.yaml",
        type=str,
        help="Path to the configuration file",
    )
    parser.add_argument(
        "--output", default="data/gen_data", type=str, help="Path to save the dataset"
    )
    parser.add_argument(
        "--samples", type=int, help="Number of combinations to sample (default: all)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=5, help="Number of concurrent API requests"
    )
    parser.add_argument(
        "--model", type=str, default="gpt-4o-mini", help="OpenAI model to use"
    )
    parser.add_argument("--system-message", type=str, help="System message for the API")
    parser.add_argument(
        "--no-intermediate", action="store_true", help="Don't save intermediate results"
    )

    args = parser.parse_args()

    # Run the generator
    asyncio.run(
        generate_dataset(
            config_file=args.config,
            output_path=args.output,
            num_samples=args.samples,
            batch_size=args.batch_size,
            openai_model=args.model,
            system_message=args.system_message,
            save_intermediate=not args.no_intermediate,
        )
    )


if __name__ == "__main__":
    main()
