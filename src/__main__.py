"""Main entry point for function calling with constrained decoding."""

import argparse
from pathlib import Path
from src.pipeline import load_json_list
from src.models import FunctionDefinition, PromptEntry


def parse_arg() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        The parsed command line arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Function calling with constrained decoding."
    )
    parser.add_argument(
        "--functions_definition",
        "--function_definition",
        type=Path,
        default=Path("data/input/functions_definition.json"),
        help="Path to function definitions JSON file",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/input/function_calling_tests.json"),
        help="Path to function calling JSON files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/output/function_calling_results.json"),
        help="Path where the output JSON will be written",
    )
    return parser.parse_args()


def main() -> None:
    """Run the function calling pipeline."""
    args = parse_arg()
    functions = load_json_list(args.functions_definition, FunctionDefinition)
    prompts = load_json_list(args.input, PromptEntry)
    print(f"Loaded {len(functions)} functions and {len(prompts)} test prompts.")


if __name__ == "__main__":
    main()