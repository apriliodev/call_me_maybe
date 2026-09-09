import argparse
from pathlib import Path
from pipeline import load_json_list
from models import FunctionDefinition, PromptEntry
def parse_arg() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
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
    args = parse_arg()
    function = load_json_list(args.function_definition, FunctionDefinition)
    prompt = load_json_list(args.input, PromptEntry)
    print(args.function_definition, args.input, args.output)

if __name__ == "__main__":
    main()