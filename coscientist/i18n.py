"""
Lightweight multilingual support (ru / en / zh + other).

Language detection uses Unicode-script heuristics (no extra dependency), which
is robust enough for routing the case corpus. Translation is delegated to an
injectable chat model (built via ``model_factory`` in production), so the whole
module is offline-testable.
"""

from __future__ import annotations

from typing import Any, Optional

SUPPORTED = ("ru", "en", "zh")
_LANG_NAMES = {"ru": "Russian", "en": "English", "zh": "Chinese"}


def _script_counts(text: str) -> dict[str, int]:
    cyrillic = latin = han = 0
    for ch in text:
        code = ord(ch)
        if 0x0400 <= code <= 0x04FF:
            cyrillic += 1
        elif (0x0041 <= code <= 0x005A) or (0x0061 <= code <= 0x007A):
            latin += 1
        elif 0x4E00 <= code <= 0x9FFF:
            han += 1
    return {"ru": cyrillic, "en": latin, "zh": han}


def detect_language(text: str) -> str:
    """Return 'ru', 'en', 'zh', or 'other' by dominant script."""
    counts = _script_counts(text or "")
    total = sum(counts.values())
    if total == 0:
        return "other"
    # Han script is information-dense, so let it win even against more Latin
    # characters (e.g. a Chinese sentence with a few Latin units/abbreviations).
    if counts["zh"] > 0 and counts["zh"] * 3 >= counts["en"]:
        return "zh"
    best = max(counts, key=counts.get)
    return best if counts[best] > 0 else "other"


def language_name(code: str) -> str:
    return _LANG_NAMES.get(code, code)


def translate(
    text: str,
    target_lang: str,
    llm: Any,
    *,
    source_lang: Optional[str] = None,
) -> str:
    """
    Translate ``text`` into ``target_lang`` using an injected chat model.

    ``llm`` must expose ``.invoke(prompt) -> obj.content`` (a langchain chat
    model in production, a fake in tests). No-op when already in target.
    """
    src = source_lang or detect_language(text)
    if src == target_lang or not text.strip():
        return text
    target_name = language_name(target_lang)
    prompt = (
        f"Translate the following text into {target_name}. Preserve technical "
        f"terms, numbers and units exactly. Return only the translation.\n\n{text}"
    )
    response = llm.invoke(prompt)
    content = getattr(response, "content", response)
    return content if isinstance(content, str) else str(content)
