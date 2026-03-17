from __future__ import annotations

"""
LLM client wrapper for the IOTAV3 backend.

For now this delegates to the existing Qwen/OpenAI integration used by
the legacy backend, but keeps a narrow interface tailored to the new
pipeline.
"""

from typing import Callable, List

from qwen_model import generate_answer as _qwen_generate_answer  # type: ignore


def _chunk_text(text: str, max_chunk_chars: int = 280) -> List[str]:
    """
    Split text into reasonably sized chunks for streaming.

    This is a simple heuristic:
    - Try to break on sentence boundaries (., ?, !, Arabic ؟ / ۔).
    - Merge short sentences together up to ~max_chunk_chars.
    - Fallback to fixed-size slicing if needed.
    """

    if not text:
        return []

    # Naive sentence split on punctuation.
    import re

    pattern = re.compile(r"(?<=[.?!؟۔])\s+")
    sentences = [s.strip() for s in pattern.split(text) if s.strip()]
    chunks: List[str] = []
    current: List[str] = []
    current_len = 0

    for sent in sentences:
        if current_len + len(sent) + 1 > max_chunk_chars and current:
            chunks.append(" ".join(current).strip())
            current = [sent]
            current_len = len(sent)
        else:
            current.append(sent)
            current_len += len(sent) + 1

    if current:
        chunks.append(" ".join(current).strip())

    # Fallback: if chunking failed somehow, just slice the string.
    if not chunks:
        text = text.strip()
        for i in range(0, len(text), max_chunk_chars):
            chunks.append(text[i : i + max_chunk_chars])

    return chunks


def generate(
    system_prompt: str,
    user_prompt: str,
    *,
    stream: bool = False,
    on_chunk: Callable[[str], None] | None = None,
) -> str:
    """
    Generate an answer from the underlying LLM (Qwen).

    This adapter currently uses the legacy `qwen_model.generate_answer`
    API. The effective prompt seen by Qwen is the concatenation of the
    IOTAV3 system/persona instructions and the user/context block:

        full_prompt = system_prompt + "\\n\\n" + user_prompt

    The `user_prompt` argument is expected to contain the CONTEXT +
    QUESTION structure produced by the IOTAV3 context engine.

    When `stream` is True and `on_chunk` is provided, the function
    generates the full answer once and then streams it back in a few
    text chunks via `on_chunk`, while also returning the full answer.
    """

    full_prompt = system_prompt.strip() + "\n\n" + user_prompt.strip()

    # Non-streaming path: one-shot generation.
    if not stream or on_chunk is None:
        return _qwen_generate_answer(context=full_prompt, user_query="")

    # Streaming path: generate once, then chunk.
    full_answer = _qwen_generate_answer(context=full_prompt, user_query="")
    for chunk in _chunk_text(full_answer):
        if chunk:
            on_chunk(chunk)
    return full_answer

