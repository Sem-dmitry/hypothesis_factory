"""
Vision-model (VLM) image parsing for the corpus.

Images in the case corpus (flotation schemes, regulations, equipment lists) are
turned into rich text descriptions by a RouterAI vision model, so they become
first-class, searchable, citable corpus entries. The model client is obtained
from ``coscientist.model_factory`` and can be injected for offline testing.
"""

from __future__ import annotations

import base64
import mimetypes
import os
from typing import Any, Optional

# Prompt tuned for engineering diagrams / tables typical of a beneficiation plant.
DEFAULT_VLM_PROMPT = (
    "You are a metallurgical process engineer. Describe this image for a technical "
    "knowledge base used to generate ore-beneficiation hypotheses. If it is a "
    "flotation/grinding flowsheet, list the equipment, streams and their order "
    "(mills, classifiers, hydrocyclones, flotation banks, cells, tanks, screens, "
    "reagent points). If it is a table or regulation, transcribe the key rows and "
    "numeric parameters. Be precise and factual; do not invent values. Answer in "
    "the dominant language of the image (Russian if the image is in Russian)."
)


def encode_image_data_url(path: str) -> str:
    """Read an image file and return a base64 ``data:`` URL."""
    mime, _ = mimetypes.guess_type(path)
    if mime is None:
        ext = os.path.splitext(path)[1].lower().lstrip(".")
        mime = f"image/{ext or 'png'}"
    with open(path, "rb") as fh:
        encoded = base64.b64encode(fh.read()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _extract_text(response: Any) -> str:
    """Pull plain text out of a langchain message or a raw string."""
    if response is None:
        return ""
    if isinstance(response, str):
        return response
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content
    # content can be a list of parts ({"type": "text", "text": ...}).
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                parts.append(part.get("text", ""))
            else:
                parts.append(str(part))
        return "".join(parts)
    return str(content)


def describe_image(
    path: str,
    *,
    prompt: str = DEFAULT_VLM_PROMPT,
    model: Optional[str] = None,
    client: Any = None,
) -> str:
    """
    Describe an image with a RouterAI vision model.

    Parameters
    ----------
    prompt : str
        Instruction sent alongside the image.
    model : str | None
        RouterAI vision model spec (defaults via ``get_vision_model``).
    client : optional
        An object with ``.invoke(messages)`` (a langchain chat model). When
        omitted, one is built lazily from ``model_factory.get_vision_model``.
        Injecting a stub makes this fully offline/testable.
    """
    if client is None:
        from coscientist.model_factory import get_vision_model

        client = get_vision_model(model)

    data_url = encode_image_data_url(path)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        }
    ]
    response = client.invoke(messages)
    return _extract_text(response).strip()
