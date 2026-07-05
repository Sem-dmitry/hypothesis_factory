"""
Domain helpers for flotation-tailings reports.

The built-in guide below is distilled from ``data/Как читать отчет института
по хвостам.docx`` so uploaded tailings spreadsheets remain interpretable even
when that explanatory DOCX is not uploaded with a run.
"""

from __future__ import annotations

import re
from typing import Iterable, Sequence


TAILINGS_GUIDE_TITLE = "Built-in guide: how to read institute tailings reports"

TAILINGS_REPORT_GUIDE = """
Tailings reports describe final waste streams from flotation/concentration
plants. Metals found in tailings are treated as losses, so tailings tables are
primary plant evidence for hypotheses about reducing metal losses.

For Nornickel-style reports:
- Tailings may be rock tailings or pyrrhotite tailings. Both are waste streams;
  pyrrhotite tailings have passed extra processing stages to reduce non-ferrous
  metal content.
- SMT means dry metric tonnes. Tables usually include processed material volume,
  generated tailings volume, and element grades/tonnes in products.
- Element 28 is nickel. Element 29 is copper.
- Particle-size classes such as +125, -125 +71, -71 +45, -45 +20, -20 +10,
  and -10 are sieve fractions in micrometres. For example, -125 +71 means
  particles smaller than 125 micrometres and larger than 71 micrometres.
- Mineralogical rows show which minerals host elements 28/29 in each size
  class. For element 28, potentially recoverable forms include open/liberated
  and locked/closed Pnt (pentlandite) and millerite. For element 29, the
  recoverable forms are open/liberated and locked/closed Pnt/Cp.
- Recommendations should connect the dominant loss fraction/mineral form to
  realistic plant levers: grind/classification, flotation residence time,
  reagent regime, pyrrhotite depression, collector/frother changes, or circuit
  settings using existing equipment and constraints.
""".strip()

TAILINGS_KEYWORDS = (
    "хвост",
    "tailing",
    "tailings",
    "элемент 28",
    "element 28",
    "элемент 29",
    "element 29",
    "содержание элемент",
    "потер",
    "loss",
    "losses",
    "смт",
    "smt",
    "класс крупности",
    "крупност",
    "pnt",
    "pentland",
    "пентланд",
    "millerite",
    "миллерит",
    "pyrrhotite",
    "пирротин",
    "pnt/cp",
    "cp",
)

TAILINGS_HEADER_HINTS = (
    "материал",
    "смт",
    "элемент 28",
    "элемент 29",
    "доля потерь",
    "класс крупности",
    "минерал",
    "pnt",
    "cp",
)


def _norm(text: str) -> str:
    return (text or "").casefold()


def is_tailings_related(*texts: str) -> bool:
    """Return True when text likely concerns tailings reports/losses."""
    joined = "\n".join(t for t in texts if t)
    low = _norm(joined)
    return any(k in low for k in TAILINGS_KEYWORDS)


def extract_domain_keywords(text: str, *, limit: int = 12) -> list[str]:
    """Extract stable tailings-domain keyword labels present in text."""
    low = _norm(text)
    found: list[str] = []
    for keyword in TAILINGS_KEYWORDS:
        if keyword in low and keyword not in found:
            found.append(keyword)
        if len(found) >= limit:
            break
    return found


def detect_header_lines(rows: Sequence[tuple[int, str]], *, max_rows: int = 8) -> list[str]:
    """
    Pick likely header/context rows from the top of a worksheet.

    Tailings spreadsheets often have merged-style multi-row headers, so we keep
    several dense/keyword-rich rows instead of assuming row 1 is enough.
    """
    selected: list[str] = []
    for _row_no, line in rows[:max_rows]:
        low = _norm(line)
        has_hint = any(h in low for h in TAILINGS_HEADER_HINTS)
        has_many_columns = line.count("|") >= 2
        if has_hint or has_many_columns:
            selected.append(line)
    return selected[:4]


def summarize_headers(lines: Iterable[str], *, max_chars: int = 700) -> str:
    text = "\n".join(line.strip() for line in lines if line and line.strip())
    text = re.sub(r"[ \t]+", " ", text).strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + " ..."


def tailings_guide_block() -> str:
    """Prompt-ready domain guide that is explicitly not corpus evidence."""
    return (
        f"# {TAILINGS_GUIDE_TITLE}\n"
        "Use this built-in guide to interpret tailings XLSX/DOCX/PDF data. "
        "It is domain guidance, not an uploaded source citation.\n"
        f"{TAILINGS_REPORT_GUIDE}"
    )


def enrich_xlsx_text(
    *,
    raw_text: str,
    source_name: str,
    sheet_name: str,
    row_start: int,
    row_end: int,
    header_lines: Sequence[str],
) -> tuple[str, dict]:
    """
    Return text/metadata for a spreadsheet chunk, with stronger enrichment for
    likely tailings reports.
    """
    header_summary = summarize_headers(header_lines)
    likely_tailings = is_tailings_related(source_name, sheet_name, header_summary, raw_text)
    metadata = {
        "sheet": sheet_name,
        "row_start": row_start,
        "row_end": row_end,
        "headers": list(header_lines),
        "tailings_report": likely_tailings,
        "domain_keywords": extract_domain_keywords(
            "\n".join([source_name, sheet_name, header_summary, raw_text])
        ),
    }

    prefix = [
        f"Spreadsheet source: {source_name}",
        f"Worksheet: {sheet_name}",
        f"Rows: {row_start}-{row_end}",
    ]
    if header_summary:
        prefix.append("Worksheet/header context:\n" + header_summary)

    if likely_tailings:
        prefix.extend(
            [
                "Domain interpretation: this appears to be a flotation tailings "
                "report. Treat these rows as primary plant evidence for metal "
                "losses to tailings, including element 28 (nickel), element 29 "
                "(copper), dry metric tonnes (SMT), particle-size fractions, "
                "and mineral forms such as Pnt/pentlandite, millerite, Cp/"
                "chalcopyrite, and pyrrhotite.",
                "When evaluating hypotheses, connect the numeric loss/mineralogy "
                "pattern in this spreadsheet to realistic process levers: grind "
                "size, classification, residence time, reagent regime, pyrrhotite "
                "depression, collector/frother changes, and existing circuit "
                "constraints.",
                tailings_guide_block(),
            ]
        )

    return "\n\n".join(prefix + ["Spreadsheet rows:\n" + raw_text]), metadata
