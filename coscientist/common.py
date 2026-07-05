import logging
import os
import re
from typing import Callable, TypeVar

from jinja2 import Environment, FileSystemLoader, select_autoescape

from coscientist.custom_types import ParsedHypothesis

# Sentinel returned by ``retry_call`` when every attempt failed.
RETRY_FAILED = object()

_T = TypeVar("_T")


def retry_call(fn: Callable[[], _T], *, label: str = "operation", attempts: int = 3):
    """
    Call ``fn`` up to ``attempts`` times, returning its result on the first
    success. If every attempt raises, log each failure and return
    :data:`RETRY_FAILED` (rather than propagating), so one flaky agent response
    never aborts a whole multi-step run. Callers check ``result is RETRY_FAILED``
    and skip / fall back accordingly.
    """
    for i in range(attempts):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - resilience boundary
            logging.warning(
                "%s attempt %d/%d failed: %s", label, i + 1, attempts, exc
            )
    logging.warning("%s failed after %d attempts; skipping.", label, attempts)
    return RETRY_FAILED

_env = Environment(
    loader=FileSystemLoader(os.path.join(os.path.dirname(__file__), "prompts")),
    autoescape=select_autoescape(),
    trim_blocks=True,
    lstrip_blocks=True,
)


# ---------------------------------------------------------------------------
# Language policy (adaptation for Russian users).
#
# Every agent prompt is rendered through ``load_prompt`` (there are no other
# template render sites). We therefore append the language policy here, once, so
# it reliably reaches EVERY current and future agent — an explicit addition to
# the prompt, not a rewrite of the prompt bodies into Russian.
#
# Two policies:
#  * LANG_POLICY_RU     — agent-to-user communication: respond in Russian; keep
#                         all structural markers / JSON keys / ALL-CAPS control
#                         tokens verbatim so parsers keep working.
#  * LANG_POLICY_SEARCH — for prompts whose output becomes WEB-SEARCH QUERIES
#                         (the Literature agent's topic decomposition): queries
#                         MAY be in English / other languages to reach world
#                         practice and international sources.
# ---------------------------------------------------------------------------

# Unique marker substrings used by tests to assert the policy is present.
_RU_POLICY_MARKER = "Отвечай на русском языке"
_SEARCH_POLICY_MARKER = "мировую практику"

LANG_POLICY_RU = (
    "\n\n---\n"
    "# Язык ответа (ОБЯЗАТЕЛЬНО)\n"
    f"{_RU_POLICY_MARKER}. Все гипотезы, рассуждения, обоснования, оценки, отчёты и любой\n"
    "текст, предназначенный для человека, пиши по-русски — даже если цель, материалы, корпус или\n"
    "сама эта инструкция написаны на английском.\n"
    "НО в ТОЧНОСТИ сохраняй требуемый формат вывода, заданный выше в этой инструкции, и НЕ переводи\n"
    "служебные элементы — оставляй их ровно как указано:\n"
    "- маркеры-заголовки markdown (например `# Hypothesis`, `### Subtopic N`);\n"
    "- ключи и названия полей JSON;\n"
    "- управляющие метки-команды в ВЕРХНЕМ РЕГИСТРЕ и их обязательные значения/токены\n"
    "  (метки, оканчивающиеся двоеточием, финальные вердикты и названия действий) — пиши их на\n"
    "  английском ровно так, как требует формат выше.\n"
    "По-русски пишется ТОЛЬКО содержательный текст внутри этих структур. Общепринятые научные и\n"
    "технические термины, химические формулы и обозначения можно оставлять как есть.\n"
)

LANG_POLICY_SEARCH = (
    "\n\n---\n"
    "# Язык подтем/поисковых запросов\n"
    "Эти подтемы используются как поисковые запросы к веб-поиску литературы. Чтобы охватить\n"
    f"{_SEARCH_POLICY_MARKER} и международные источники (зарубежные статьи, патенты, отраслевые\n"
    "кейсы заводов), можно и НУЖНО формулировать подтемы с использованием общепринятой английской\n"
    "терминологии: пиши подтему полностью на английском ИЛИ добавляй англоязычные ключевые термины\n"
    "и синонимы рядом с русскими. Не ограничивайся только русскоязычными формулировками — цель\n"
    "запросов в том, чтобы найти мировой опыт и подтверждения по теме. Служебные заголовки\n"
    "`### Subtopic N` оставляй на английском.\n"
)

# Prompts whose model output is turned into web-search queries.
_SEARCH_QUERY_PROMPTS = frozenset({"topic_decomposition"})


def language_policy_for(name: str) -> str:
    """Return the language-policy suffix appended to prompt ``name``."""
    return LANG_POLICY_SEARCH if name in _SEARCH_QUERY_PROMPTS else LANG_POLICY_RU


def load_prompt(name: str, **kwargs) -> str:
    """
    Load a template from the prompts directory and render it with the given
    kwargs, then append the language policy (Russian output for agents; a
    multilingual-queries policy for search-query prompts).

    Parameters
    ----------
    name: str
        The name of the template to load, without the .md extension.
    **kwargs: dict
        The kwargs to render the template with.

    Returns
    -------
    str
        The rendered template with the language policy appended.
    """
    rendered = _env.get_template(f"{name}.md").render(**kwargs)
    return rendered + language_policy_for(name)


# Heading synonyms (English + Russian) so a model that localizes the section
# titles still parses. Order matters: check the most specific first.
_SECTION_SYNONYMS = {
    "hypothesis": ["hypothesis", "гипотеза", "гипотез"],
    "predictions": [
        "falsifiable prediction", "prediction", "предсказан", "проверяем", "прогноз"
    ],
    "assumptions": ["assumption", "допущен", "предположен", "предпосылк"],
}


def _classify_heading(title: str) -> str | None:
    title = title.strip().lower()
    for kind, synonyms in _SECTION_SYNONYMS.items():
        if any(s in title for s in synonyms):
            return kind
    return None


def parse_hypothesis_markdown(markdown_text: str) -> ParsedHypothesis:
    """
    Parse a model's markdown into a :class:`ParsedHypothesis`.

    Tolerant by design: accepts English or localized (Russian) headings, any
    heading level (``#``–``####``), inline content (``# Hypothesis: ...``) and
    code-fenced output. Only the hypothesis is strictly required; predictions
    and assumptions default to empty rather than crashing the whole run when a
    model omits or mis-formats them.
    """
    text = markdown_text or ""
    if "#FINAL REPORT#" in text:
        text = text.split("#FINAL REPORT#", 1)[1]

    # Strip a surrounding code fence, if present.
    fence = re.match(r"^\s*```[a-zA-Z]*\s*\n(.*)\n```\s*$", text, re.DOTALL)
    if fence:
        text = fence.group(1)

    # Locate markdown headings (any level) and slice out each section body.
    heading_re = re.compile(r"^\s{0,3}#{1,4}\s*(.+?)\s*$", re.MULTILINE)
    matches = list(heading_re.finditer(text))

    hypothesis = ""
    predictions: list[str] = []
    assumptions: list[str] = []

    for i, hm in enumerate(matches):
        title = hm.group(1)
        body_start = hm.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[body_start:body_end].strip()

        # Support inline content: "Hypothesis: the actual hypothesis".
        if ":" in title:
            head, _, inline = title.partition(":")
            if _classify_heading(head) is not None:
                title = head
                body = (inline.strip() + ("\n" + body if body else "")).strip()

        kind = _classify_heading(title)
        if kind == "hypothesis" and not hypothesis:
            hypothesis = body
        elif kind == "predictions" and not predictions:
            predictions = _parse_numbered_list(body)
        elif kind == "assumptions" and not assumptions:
            assumptions = _parse_numbered_list(body)

    # Fallback: no recognizable hypothesis heading — use the whole text (minus
    # headings) so a well-formed idea in the wrong format still survives.
    if not hypothesis:
        stripped = heading_re.sub("", text).strip()
        hypothesis = stripped or text.strip()

    if not hypothesis:
        raise ValueError(
            f"Could not parse a hypothesis from the model output: {markdown_text[:300]}"
        )

    return ParsedHypothesis(
        hypothesis=hypothesis, predictions=predictions, assumptions=assumptions
    )


def _parse_numbered_list(content: str) -> list[str]:
    """
    Parse a numbered list from text content into a list of strings.

    Parameters
    ----------
    content : str
        Text containing a numbered list (e.g., "1. First item\n2. Second item")

    Returns
    -------
    list[str]
        List of individual items with numbering removed
    """
    if not content.strip():
        return []

    lines = content.split("\n")
    items = []

    # Regex to match various numbering formats: 1., 1), 1-, etc.
    number_pattern = re.compile(r"^\s*\d+[\.\)\-]\s*(.+)", re.MULTILINE)

    current_item = ""

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Check if line starts with a number
        match = number_pattern.match(line)
        if match:
            # If we have a current item, add it to the list
            if current_item:
                items.append(current_item.strip())
            # Start new item
            current_item = match.group(1)
        else:
            # This line is a continuation of the current item
            if current_item:
                current_item += " " + line
            else:
                # Handle case where first line doesn't start with a number
                current_item = line

    # Add the last item
    if current_item:
        items.append(current_item.strip())

    return items
