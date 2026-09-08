import argparse
from pathlib import Path


def parse_arg() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--function_definition",
        type=Path
        default=Path("data/input/function_definitions.json")
        help="Path to function definitions JSON file"
    )
    parser.add_argument(
        "--input",
        type=Path
        default=Path("data/input/function_calling_tests.json")
    )