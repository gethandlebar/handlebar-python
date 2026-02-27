"""Token counting helpers — mirrors packages/core/src/tokens.ts."""

from __future__ import annotations

_encoder = None


def _get_encoder():
    global _encoder
    if _encoder is None:
        import tiktoken

        _encoder = tiktoken.get_encoding("cl100k_base")
    return _encoder


def tokenise_count(text: str) -> int:
    """Count the number of tokens in ``text`` using the cl100k_base encoding.

    Uses the same tiktoken library as the JS implementation.
    The encoder is lazily initialised and cached.
    """
    return len(_get_encoder().encode(text))
