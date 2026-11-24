"""Vocabulary loading and management for constrained decoding."""

import json
from pathlib import Path
from llm_sdk import Small_LLM_Model


def load_vocabulary(llm: Small_LLM_Model) -> dict[str, int]:
    """Load the token-to-id vocabulary mapping from the tokenizer file.

    Args:
        llm: The Small_LLM_Model instance.

    Returns:
        A dictionary mapping token strings to token IDs.
    """
    tokenizer_path = Path(llm.get_path_to_tokenizer_file())
    with open(tokenizer_path, encoding="utf-8") as f:
        data = json.load(f)
    vocab: dict[str, int] = data["model"]["vocab"]
    return vocab


def get_numeric_tokens(vocab: dict[str,int]) -> list[int]:
    numeric_ids: list[int] = []

    for token_str, token_id in vocab.items():
        if token_str.isdigit() or token_str in (".", "-"):
            numeric_ids.append(token_id)
    return numeric_ids

