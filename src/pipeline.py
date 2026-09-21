"""Pipeline utilities for loading data and building prompts."""

import json
from pathlib import Path
from typing import TypeVar
from pydantic import BaseModel, ValidationError
from src.models import FunctionDefinition

T = TypeVar("T", bound=BaseModel)


def load_json_list(path: Path, model: type[T]) -> list[T]:
    """Load and validate a list of models from a JSON file.

    Args:
        path: Path to the JSON file.
        model: Pydantic model class to validate each entry.

    Returns:
        A list of validated model instances.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the JSON is invalid or fails schema validation.
    """
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {path}: {e}") from e

    if not isinstance(raw, list):
        raise ValueError(f"Expected a JSON list in {path}, got {type(raw).__name__}")

    try:
        return [model.model_validate(item) for item in raw]
    except ValidationError as e:
        raise ValueError(f"Schema error in {path}: {e}") from e


def format_functions_prompt(functions: list[FunctionDefinition]) -> str:
    """Build the system context listing available functions and descriptions.

    Args:
        functions: List of FunctionDefinition objects.

    Returns:
        A formatted string describing available functions.
    """
    prompt = "Available functions:\n"
    for fn in functions:
        prompt += f"- {fn.name}: {fn.description}\n"
    return prompt
