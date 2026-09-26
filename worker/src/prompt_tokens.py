"""Prompt token-limit enforcement (defence in depth behind the gateway's check).

Upstream tokenizes with truncation=True, max_length=256 and silently drops the rest of an
overlong prompt. The service must reject instead (HANDOFF section 8). The count comes
from the backend's own tokenizer call so it is exact for the path that will actually run:
the PyTorch reference path uses transformers' AutoTokenizer, the TensorRT path uses the
raw `tokenizers` library, and the two are not assumed to agree on special tokens.
"""
from __future__ import annotations

from typing import Callable

from backend import MAX_PROMPT_TOKENS, WorkerFailure

MAX_PROMPT_CODE_POINTS = 2000


def check_prompt(prompt: str, count_tokens: Callable[[str], int],
                 limit: int = MAX_PROMPT_TOKENS) -> int:
    """Return the token count, or raise WorkerFailure('validation', retryable=False)."""
    if not isinstance(prompt, str) or not prompt.strip():
        raise WorkerFailure("validation", "prompt is empty", retryable=False)
    if len(prompt) > MAX_PROMPT_CODE_POINTS:
        raise WorkerFailure("validation", "prompt exceeds the character limit", retryable=False)
    try:
        count = int(count_tokens(prompt))
    except WorkerFailure:
        raise
    except Exception:
        raise WorkerFailure("internal", "prompt tokenizer failed") from None
    if count > limit:
        raise WorkerFailure("validation", f"prompt exceeds the {limit}-token limit", retryable=False)
    return count


class TokenizerJsonCounter:
    """Counts tokens with a tokenizer.json via the `tokenizers` library.

    add_special_tokens=True mirrors `Tokenizer.encode(text)` as the upstream TensorRT
    FastTokenizer calls it. The PyTorch backend does not use this class; it counts with the
    exact transformers tokenizer object the conditioner holds.
    """

    def __init__(self, tokenizer_json: str, add_special_tokens: bool = True):
        from tokenizers import Tokenizer

        self._tokenizer = Tokenizer.from_file(tokenizer_json)
        self._add_special_tokens = add_special_tokens

    def __call__(self, prompt: str) -> int:
        return len(self._tokenizer.encode(prompt, add_special_tokens=self._add_special_tokens).ids)
