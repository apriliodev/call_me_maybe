"""Token generation using constrained decoding."""

import numpy as np
from llm_sdk import Small_LLM_Model
from src.constrained_decoder import apply_mask, get_valid_tokens_for_candidates


def generate_constrained_choice(
    llm: Small_LLM_Model,
    prompt: str,
    candidates: list[str],
    vocab: dict[str, int],
) -> str:
    """Generate a string constrained to be one of the candidate strings.

    Args:
        llm: The Small_LLM_Model instance.
        prompt: The base prompt text provided to the model.
        candidates: Allowed target candidate strings.
        vocab: Token-to-id mapping dictionary.

    Returns:
        The generated candidate string selected by the model.
    """
    generated_text = ""
    remaining_candidates = list(candidates)

    while len(remaining_candidates) > 1 or generated_text not in remaining_candidates:
        input_ids_tensor = llm.encode(prompt + generated_text)
        input_ids: list[int] = input_ids_tensor[0].tolist()
        raw_logits = np.array(llm.get_logits_from_input_ids(
            input_ids), dtype=np.float64)

        valid_tokens = get_valid_tokens_for_candidates(
            remaining_candidates, generated_text, vocab
        )
        if not valid_tokens:
            break

        masked_logits = apply_mask(raw_logits, valid_tokens)
        best_id = int(np.argmax(masked_logits))
        token_str = str(llm.decode([best_id]))

        generated_text += token_str
        remaining_candidates = [
            c for c in remaining_candidates if c.startswith(generated_text)
        ]

        if generated_text in remaining_candidates:
            return str(generated_text)

    return str(generated_text)


def generate_constrained_number(llm: Small_LLM_Model, prompt: str, numeric_tokens: list[int], stop_token_id: int, vocab: dict[str, int]) -> float | int:
    generated_text: str = ""
    stop_id: list[int] = []
    stop_id.append(stop_token_id)
    dot_id = vocab["."]
    hyph_id = vocab["-"]
    for _ in range(20):
        input_ids = llm.encode(prompt + generated_text)[0].tolist()
        raw_logits = np.array(llm.get_logits_from_input_ids(input_ids))
        if generated_text == "":
            allowed_tokens =  [token_id for token_id in numeric_tokens if token_id != dot_id]
        else:
            allowed_tokens = numeric_tokens + stop_id
            allowed_tokens.remove(hyph_id)
            if "." in generated_text:
                allowed_tokens =  [token_id for token_id in allowed_tokens if token_id != dot_id]
        masked_logits = apply_mask(raw_logits, allowed_tokens)
        best_id = int(np.argmax(masked_logits))
        if best_id == stop_token_id:
            break
        token_str = str(llm.decode([best_id]))
        generated_text += token_str
    if generated_text == "":
        raise ValueError("The model produced nothing valid")
    return float(generated_text)


def generate_constrained_string(llm: Small_LLM_Model, prompt: str, litteral_tokens: list[int], stop_token_id: int, vocab: dict[str, int]) -> str:
    generated_text = ""
    for _ in range(20):
