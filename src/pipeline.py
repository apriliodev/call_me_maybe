import json

from .models import FunctionDefinition, PromptEntry
from pathlib import Path
from pydantic import TypeAdapter, ValidationError

def load_functions_definition(path: Path) -> list[FunctionDefinition]:
    if not path.exists():
        raise FileNotFoundError(f"Functions definition file not found: {path}")
    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc
    try:
        return TypeAdapter(list[FunctionDefinition]).validate_python(raw)
    except ValidationError as exc:
        raise ValueError(f"Schema error in {path}: {exc}") from exc


def load_prompts(path: Path) -> list[PromptEntry]:
    if not path.exists():
        raise FileNotFoundError(f"Input prompts file not found: {path}")
    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc
    try:
        return TypeAdapter(list[PromptEntry]).validate_python(raw)
    except ValidationError as exc:
        raise ValueError(f"Schema error in {path}: {exc}") from exc