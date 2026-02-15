"""Translation for answer language enforcement: translate English answer to Arabic when query was Arabic."""
from __future__ import annotations


def translate_to_arabic(text: str) -> str | None:
    """
    Translate English text to Arabic. Uses OpenAI chat when OPENAI_API_KEY is set.
    Returns translated string or None on failure/disabled.
    """
    if not text or not text.strip():
        return None
    try:
        from config import ENABLE_TRANSLATE_BACK
        if not ENABLE_TRANSLATE_BACK:
            return None
    except ImportError:
        return None
    try:
        from openai import OpenAI
        client = OpenAI()
        r = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Translate the following English regulatory text to Arabic. Preserve (Page X) and (Pages X–Y) citations exactly. Output only the translation."},
                {"role": "user", "content": text.strip()},
            ],
            max_tokens=800,
        )
        if r.choices and r.choices[0].message and r.choices[0].message.content:
            return r.choices[0].message.content.strip() or None
    except Exception:
        pass
    return None
