"""Constrained decoding functions: logit masking and token filtering."""

import numpy as np
from numpy.typing import NDArray


def get_valid_tokens_for_candidates(
    candidates: list[str],
    generated_text: str,
    vocab: dict[str, int],
) -> list[int]:
    """Find token IDs that extend generated_text towards one of the candidates.

    Args:
        candidates: List of valid target strings.
        generated_text: The string generated so far.
        vocab: Mapping of token string to token ID.

    Returns:
        List of allowed token IDs.
    """
    tokens_ok: list[int] = []
    for token_str, token_id in vocab.items():
        new_text = generated_text + token_str
        for candidate in candidates:
            if candidate.startswith(new_text):
                tokens_ok.append(token_id)
                break
    return tokens_ok


def apply_mask(
    logits: NDArray[np.float64],
    valid_token_ids: list[int],
) -> NDArray[np.float64]:
    """Apply -infinity mask to logits for all invalid tokens.

    Args:
        logits: Raw logits array from the model.
        valid_token_ids: List of authorized token IDs.

    Returns:
        Masked logits array where unauthorized tokens have value -inf.
    """
    masked: NDArray[np.float64] = np.full_like(logits, -np.inf)
    if valid_token_ids:
        masked[valid_token_ids] = logits[valid_token_ids]
    return masked

