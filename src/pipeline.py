import json

from pathlib import Path
from pydantic import TypeAdapter, ValidationError, BaseModel
from typing import TypeVar

T = TypeVar("T", bound=BaseModel)

def load_json_list(path: Path, model:type[T]) -> list[T]:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {path}: {e}") from e
    try:
        return TypeAdapter(list[model]).validate_python(raw)
    except ValidationError as e:
        raise ValueError(f"Schema error in {path}: {e}") from e
