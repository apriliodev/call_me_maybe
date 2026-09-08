import argparse
from pathlib import Path


def parse_arg() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--function_definition",
        type=Path,
        default=Path("data/input/function_definitions.json"),
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
    args = parse_arg()
    print(args.function_definition, args.input, args.output)

if __name__ == "__main__":
    main()